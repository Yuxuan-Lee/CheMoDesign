"""Coordinate losses, including radius of gyration and helix content."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F

from .base import BaseLoss
from .utils import (
    extract_binder_ca_atoms,
    compute_rg,
    compute_termini_distance,
    compute_secondary_structure_features,
)


# =====================================================

# =====================================================

class RGLoss(BaseLoss):
    """RGLoss."""
    
    def __init__(
        self,
        weight: float = 1.0,
        target_mode: str = 'empirical',  
        target_rg: float = 12.0,  
        rg_coefficient: float = 2.38,  
        rg_exponent: float = 0.365,  
        tolerance: float = 1.5,  
        name: str = "rg_coords"
    ):
        """  init  ."""
        super().__init__(weight=weight, name=name)
        self.target_mode = target_mode
        self.target_rg = target_rg
        self.rg_coefficient = rg_coefficient
        self.rg_exponent = rg_exponent
        self.tolerance = tolerance  
    
    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        feats: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        if coords is None:
            raise ValueError("RGLoss requires coords (Diffusion output)")
        
        if feats is None or metadata is None:
            raise ValueError("RGLoss requires feats and metadata")
        
        device = coords.device
        
        
        try:
            binder_ca = extract_binder_ca_atoms(coords, feats, metadata)  # [L, 3]
        except Exception as e:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'error': f'Failed to extract binder CA: {str(e)}'
            }
        
        
        if not hasattr(self, '_rg_verified'):
            print(f"\n[DEBUG] RGLoss Verification:")
            print(f"  Total coords shape: {coords.shape}")
            binder_slice = metadata.get('binder_slice', None)
            receptor_slice = metadata.get('receptor_slice', None)
            binder_length = metadata.get('binder_length', None)
            receptor_length = metadata.get('receptor_length', None)
            if binder_slice:
                print(f"  binder_slice: {binder_slice} (tokens {binder_slice.start} to {binder_slice.stop})")
            if receptor_slice:
                print(f"  receptor_slice: {receptor_slice} (tokens {receptor_slice.start} to {receptor_slice.stop})")
            print(f"  Extracted binder_ca shape: {binder_ca.shape}")
            print(f"  Expected binder_length: {binder_length}")
            if binder_ca.shape[0] != binder_length:
                print(f" WARNING: Extracted CA count ({binder_ca.shape[0]}) != binder_length ({binder_length})")
            else:
                print(f" Extracted CA count matches binder_length")
            
            
            if receptor_slice and binder_slice:
                
                from .utils import extract_ca_atoms_by_chain
                try:
                    receptor_ca = extract_ca_atoms_by_chain(coords, feats, chain_id=1, length=receptor_length or 114, metadata=metadata)
                    receptor_ca_squeezed = receptor_ca.squeeze(0)
                    binder_center = binder_ca.mean(dim=0).detach()
                    receptor_center = receptor_ca_squeezed.mean(dim=0).detach()
                    center_dist = torch.norm(binder_center - receptor_center).item()
                    print(f"  Binder center: {binder_center.cpu().numpy()}")
                    print(f"  Receptor center: {receptor_center.cpu().numpy()}")
                    print(f"  Center-to-center distance: {center_dist:.2f} Å")
                    if center_dist < 5.0:
                        print(f" WARNING: Binder and receptor centers are very close ({center_dist:.2f} Å), may indicate extraction error!")
                except Exception as e:
                    print(f" Could not verify receptor separation: {e}")
            
            self._rg_verified = True
        
        L = binder_ca.shape[0]
        
        
        rg = compute_rg(binder_ca)  
        
        
        if self.target_mode == 'empirical':
            
            target_rg = self.rg_coefficient * (L ** self.rg_exponent)
        elif self.target_mode == 'fixed':
            
            target_rg = self.target_rg
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")
        
        
        
        # 
        
        
        
        # 
        
        
        
        
        # 
        
        
        #   RG < 7.0 Å -> Loss = smooth_l1(RG, 7.0)
        #   RG > 10.0 Å -> Loss = smooth_l1(RG, 10.0)
        
        lower_bound = target_rg - self.tolerance
        upper_bound = target_rg + self.tolerance
        
        if rg.item() < lower_bound:
            
            target_value = lower_bound
            raw_loss = F.smooth_l1_loss(
                rg.unsqueeze(0), 
                torch.tensor([target_value], device=device),
                beta=1.0,  
                reduction='mean'
            )
        elif rg.item() > upper_bound:
            
            target_value = upper_bound
            raw_loss = F.smooth_l1_loss(
                rg.unsqueeze(0), 
                torch.tensor([target_value], device=device),
                beta=1.0,
                reduction='mean'
            )
        else:
            
            
            
            deviation = (rg - target_rg) / self.tolerance  
            raw_loss = 0.01 * (deviation ** 2)  
        
        
        weighted_loss = self.weight * raw_loss
        
        
        loss_info = {
            'raw_loss': raw_loss.item(),
            'weighted_loss': weighted_loss.item(),
            'rg_value': rg.item(),
            'target_rg': target_rg,
            'rg_diff': (rg - target_rg).item(),
            'lower_bound': lower_bound,  
            'upper_bound': upper_bound,  
            'in_dead_zone': lower_bound <= rg.item() <= upper_bound,  
            'binder_length': L,
            'target_mode': self.target_mode,
        }
        
        return weighted_loss, loss_info


# =====================================================

# =====================================================

class HelixLossCoords(BaseLoss):
    """HelixLossCoords."""
    
    def __init__(
        self,
        weight: float = 1.0,
        offset: int = 3,  
        target_distance: float = 5.5,  
        tolerance: float = 1.0,  
        satisfaction_threshold: float = 0.7,  
        name: str = "helix_coords"
    ):
        """  init  ."""
        super().__init__(weight=weight, name=name)
        self.offset = offset
        self.target_distance = target_distance
        self.tolerance = tolerance
        self.satisfaction_threshold = satisfaction_threshold  
    
    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        feats: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        if coords is None:
            raise ValueError("HelixLossCoords requires coords (Diffusion output)")
        
        if feats is None or metadata is None:
            raise ValueError("HelixLossCoords requires feats and metadata")
        
        device = coords.device
        
        
        try:
            binder_ca = extract_binder_ca_atoms(coords, feats, metadata)  # [L, 3]
        except Exception as e:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'error': f'Failed to extract binder CA: {str(e)}'
            }
        
        
        if not hasattr(self, '_helix_verified'):
            print(f"\n[DEBUG] HelixLossCoords Verification:")
            print(f"  Total coords shape: {coords.shape}")
            binder_slice = metadata.get('binder_slice', None)
            binder_length = metadata.get('binder_length', None)
            if binder_slice:
                print(f"  binder_slice: {binder_slice} (tokens {binder_slice.start} to {binder_slice.stop})")
            print(f"  Extracted binder_ca shape: {binder_ca.shape}")
            print(f"  Expected binder_length: {binder_length}")
            if binder_ca.shape[0] != binder_length:
                print(f" WARNING: Extracted CA count ({binder_ca.shape[0]}) != binder_length ({binder_length})")
            else:
                print(f" Extracted CA count matches binder_length")
            self._helix_verified = True
        
        L = binder_ca.shape[0]
        
        
        if L < self.offset + 1:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'reason': f'binder_length ({L}) < offset+1 ({self.offset+1})'
            }
        
        
        i_coords = binder_ca[:-self.offset]  # [L-offset, 3]
        j_coords = binder_ca[self.offset:]  # [L-offset, 3]
        
        offset_distances = torch.sqrt(
            ((i_coords - j_coords) ** 2).sum(dim=-1) + 1e-8
        )  # [L-offset]
        
        
        target = torch.full_like(offset_distances, self.target_distance)
        raw_loss = F.smooth_l1_loss(offset_distances, target, reduction='mean')
        
        
        
        satisfaction_tolerance = 1.5
        satisfied_pairs = (torch.abs(offset_distances - self.target_distance) <= satisfaction_tolerance).float()
        satisfaction_ratio = satisfied_pairs.mean().item()  
        
        
        
        
        if satisfaction_ratio >= self.satisfaction_threshold:
            
            adaptive_weight = self.weight * 0.01
            is_satisfied = True
        elif satisfaction_ratio >= (self.satisfaction_threshold - 0.2):
            
            
            t = (satisfaction_ratio - (self.satisfaction_threshold - 0.2)) / 0.2  # [0, 1]
            adaptive_weight = self.weight * (1.0 * (1 - t) + 0.01 * t)
            is_satisfied = False
        else:
            
            adaptive_weight = self.weight
            is_satisfied = False
        
        
        weighted_loss = adaptive_weight * raw_loss
        
        
        loss_info = {
            'raw_loss': raw_loss.item(),
            'weighted_loss': weighted_loss.item(),
            'offset': self.offset,
            'target_distance': self.target_distance,
            'mean_offset_distance': offset_distances.mean().item(),
            'std_offset_distance': offset_distances.std().item(),
            'num_pairs': len(offset_distances),
            'binder_length': L,
            'satisfaction_ratio': satisfaction_ratio,  
            'satisfaction_threshold': self.satisfaction_threshold,  
            'is_satisfied': is_satisfied,  
            'adaptive_weight_ratio': adaptive_weight / self.weight,  
        }
        
        return weighted_loss, loss_info


# =====================================================

# =====================================================

class TerminiDistanceLoss(BaseLoss):
    """TerminiDistanceLoss."""
    
    def __init__(
        self,
        weight: float = 1.0,
        threshold: float = 7.0,  
        name: str = "termini_distance"
    ):
        """  init  ."""
        super().__init__(weight=weight, name=name)
        self.threshold = threshold
    
    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        feats: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        if coords is None:
            raise ValueError("TerminiDistanceLoss requires coords (Diffusion output)")
        
        if feats is None or metadata is None:
            raise ValueError("TerminiDistanceLoss requires feats and metadata")
        
        device = coords.device
        
        
        is_cyclic = metadata.get('is_cyclic', False)
        if not is_cyclic:
            
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'reason': 'not cyclic peptide'
            }
        
        
        try:
            binder_ca = extract_binder_ca_atoms(coords, feats, metadata)  # [L, 3]
        except Exception as e:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'raw_loss': 0.0,
                'weighted_loss': 0.0,
                'error': f'Failed to extract binder CA: {str(e)}'
            }
        
        
        termini_distance = compute_termini_distance(binder_ca)  
        
        
        
        diff = termini_distance - self.threshold
        raw_loss = F.elu(diff)  
        
        
        weighted_loss = self.weight * raw_loss
        
        
        loss_info = {
            'raw_loss': raw_loss.item(),
            'weighted_loss': weighted_loss.item(),
            'termini_distance': termini_distance.item(),
            'threshold': self.threshold,
            'over_threshold': termini_distance.item() > self.threshold,
        }
        
        return weighted_loss, loss_info


# =====================================================

# =====================================================

class ClashLoss(BaseLoss):
    """ClashLoss."""
    
    def __init__(self, weight: float = 1.0, name: str = "clash"):
        super().__init__(weight=weight, name=name)
        raise NotImplementedError("ClashLoss is not implemented yet (Phase 2)")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ClashLoss is not implemented yet (Phase 2)")


# =====================================================

# =====================================================

class ContactLossCoords(BaseLoss):
    """ContactLossCoords."""
    
    def __init__(self, weight: float = 1.0, name: str = "contact_coords"):
        super().__init__(weight=weight, name=name)
        raise NotImplementedError("ContactLossCoords is not implemented yet (Phase 2)")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ContactLossCoords is not implemented yet (Phase 2)")

