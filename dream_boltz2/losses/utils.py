"""Geometry helpers for backbone losses."""

from typing import Dict, Optional, Tuple
import torch


# =====================================================

# =====================================================

def extract_ca_atoms(
    coords: torch.Tensor,
    feats: Dict,
    token_slice: Optional[slice] = None,
    length: Optional[int] = None,
    fallback_mode: str = '14_atoms_per_residue'
) -> torch.Tensor:
    """extract ca atoms."""
    
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)  # [1, N_atoms, 3]
    
    device = coords.device
    
    
    if 'token_to_center_atom' in feats and feats['token_to_center_atom'] is not None:
        token_to_center = feats['token_to_center_atom']  # [B, N_tokens, N_atoms] (one-hot) or [B, N_tokens]
        
        
        if token_slice is not None:
            token_indices = torch.arange(
                token_slice.start, token_slice.stop,
                device=device, dtype=torch.long
            )
        else:
            
            num_tokens = token_to_center.shape[1]
            token_indices = torch.arange(0, num_tokens, device=device, dtype=torch.long)
        
        
        if token_to_center.dim() == 3:
            
            token_to_center_slice = token_to_center[0, token_indices]  # [L, N_atoms]
            ca_indices = token_to_center_slice.argmax(dim=-1)  # [L]
        elif token_to_center.dim() == 2:
            
            ca_indices = token_to_center[0, token_indices]  # [L]
        else:
            raise ValueError(f"Unsupported token_to_center shape: {token_to_center.shape}")
        
        
        ca_indices = ca_indices.clamp(max=coords.shape[1] - 1)
    
    else:
        
        if length is None:
            raise ValueError("length must be provided when token_to_center_atom is not available")
        
        if fallback_mode == '14_atoms_per_residue':
            ca_indices = torch.arange(0, length, device=device) * 14 + 1
            ca_indices = ca_indices.clamp(max=coords.shape[1] - 1)
        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")
    
    
    ca_coords = coords[0, ca_indices.long(), :]  # [L, 3]
    
    return ca_coords


def extract_binder_ca_atoms(
    coords: torch.Tensor,
    feats: Dict,
    metadata: Dict
) -> torch.Tensor:
    """extract binder ca atoms."""
    binder_slice = metadata.get('binder_slice', None)
    binder_length = metadata.get('binder_length', None)
    
    if binder_slice is None or binder_length is None:
        raise ValueError("metadata must contain 'binder_slice' and 'binder_length'")
    
    return extract_ca_atoms(
        coords, feats,
        token_slice=binder_slice,
        length=binder_length
    )


# =====================================================

# =====================================================

def compute_rg(ca_coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """compute rg."""
    centroid = ca_coords.mean(dim=0)  # [3]
    squared_distances = ((ca_coords - centroid) ** 2).sum(dim=-1)  # [L]
    rg = torch.sqrt(squared_distances.mean() + eps)  
    return rg


def extract_ca_atoms_by_chain(
    coords: torch.Tensor,
    feats: dict,
    chain_id: int,
    length: int,
    metadata: dict = None  
) -> torch.Tensor:
    """extract ca atoms by chain."""
    
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)  # [1, N_atoms, 3]
    
    
    
    if chain_id == 0:
        
        if metadata is not None and 'binder_slice' in metadata:
            binder_slice = metadata['binder_slice']
            start_token = binder_slice.start
            end_token = binder_slice.stop
        else:
            
            start_token = 0
            end_token = length
    elif chain_id == 1:
        
        if metadata is not None and 'receptor_slice' in metadata:
            receptor_slice = metadata['receptor_slice']
            start_token = receptor_slice.start
            end_token = receptor_slice.stop
        elif metadata is not None and 'binder_length' in metadata:
            
            binder_length = metadata['binder_length']
            start_token = binder_length
            end_token = binder_length + length
        else:
            raise ValueError("For chain_id=1, metadata must contain 'receptor_slice' or 'binder_length'")
    else:
        raise ValueError(f"Unsupported chain_id: {chain_id} (only 0 and 1 supported)")
    
    
    
    if 'token_to_center_atom' in feats and feats['token_to_center_atom'] is not None:
        token_to_center = feats['token_to_center_atom']  # [B, N_tokens, N_atoms]
        
        
        token_indices = torch.arange(
            start_token, end_token,
            device=coords.device, dtype=torch.long
        )
        
        if token_to_center.dim() == 3:
            
            
            chain_token_to_center = token_to_center[0, token_indices]  # [length, N_atoms]
            ca_indices = chain_token_to_center.argmax(dim=-1)  # [length]
        elif token_to_center.dim() == 2:
            
            ca_indices = token_to_center[0, token_indices]  # [length]
        else:
            raise ValueError(f"Unexpected token_to_center shape: {token_to_center.shape}")
        
        
        ca_indices = ca_indices.clamp(max=coords.shape[1] - 1)
    
    
    else:
        
        ca_indices = torch.arange(
            start_token, end_token,
            device=coords.device
        ) * 14 + 1
        ca_indices = ca_indices.clamp(max=coords.shape[1] - 1)
    
    
    
    ca_coords = coords[0, ca_indices.long(), :]  # [length, 3]
    ca_coords = ca_coords.unsqueeze(0)  # [1, length, 3]
    
    return ca_coords


def compute_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """compute pairwise distances."""
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # [L, L, 3]
    dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [L, L]
    return dist_matrix


def compute_termini_distance(ca_coords: torch.Tensor) -> torch.Tensor:
    """compute termini distance."""
    n_terminus = ca_coords[0]   # [3]
    c_terminus = ca_coords[-1]  # [3]
    distance = torch.sqrt(((n_terminus - c_terminus) ** 2).sum() + 1e-8)  
    return distance


def compute_secondary_structure_features(ca_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    """compute secondary structure features."""
    L = ca_coords.shape[0]
    
    
    ca_distances = compute_pairwise_distances(ca_coords)  # [L, L]
    
    
    consecutive_distances = torch.sqrt(
        ((ca_coords[1:] - ca_coords[:-1]) ** 2).sum(dim=-1) + 1e-8
    )  # [L-1]
    
    
    if L >= 4:
        i_to_i_plus_3 = torch.sqrt(
            ((ca_coords[:-3] - ca_coords[3:]) ** 2).sum(dim=-1) + 1e-8
        )  # [L-3]
    else:
        i_to_i_plus_3 = torch.tensor([], device=ca_coords.device)
    
    
    if L >= 5:
        i_to_i_plus_4 = torch.sqrt(
            ((ca_coords[:-4] - ca_coords[4:]) ** 2).sum(dim=-1) + 1e-8
        )  # [L-4]
    else:
        i_to_i_plus_4 = torch.tensor([], device=ca_coords.device)
    
    return {
        'ca_distances': ca_distances,
        'consecutive_distances': consecutive_distances,
        'i_to_i+3_distances': i_to_i_plus_3,
        'i_to_i+4_distances': i_to_i_plus_4,
    }


# =====================================================

# =====================================================

def extract_distogram(z: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    """extract distogram."""
    
    
    if z.shape[-1] < num_bins:
        raise ValueError(f"z.shape[-1] ({z.shape[-1]}) < num_bins ({num_bins})")
    
    distogram = z[..., :num_bins]  # [B, L, L, num_bins]
    
    
    distogram = torch.softmax(distogram, dim=-1)
    
    return distogram


def distogram_to_distances(distogram: torch.Tensor, bin_edges: Optional[torch.Tensor] = None) -> torch.Tensor:
    """distogram to distances."""
    num_bins = distogram.shape[-1]
    device = distogram.device
    
    if bin_edges is None:
        
        bin_edges = torch.linspace(2.0, 22.0, num_bins + 1, device=device)
    
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # [num_bins]
    
    
    distances = (distogram * bin_centers.view(1, 1, 1, -1)).sum(dim=-1)  # [B, L, L]
    
    return distances


# =====================================================

# =====================================================

def extract_contact_prob(
    z: torch.Tensor,
    distogram: Optional[torch.Tensor] = None,
    contact_threshold: float = 8.0
) -> torch.Tensor:
    """extract contact prob."""
    if distogram is None:
        distogram = extract_distogram(z)  # [B, L, L, num_bins]
    
    num_bins = distogram.shape[-1]
    device = distogram.device
    
    
    bin_edges = torch.linspace(2.0, 22.0, num_bins + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # [num_bins]
    
    
    contact_mask = bin_centers < contact_threshold  # [num_bins]
    contact_prob = (distogram * contact_mask.view(1, 1, 1, -1)).sum(dim=-1)  # [B, L, L]
    
    return contact_prob

