#!/usr/bin/env python3
"""Interface losses, including hotspot contacts."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .base import BaseLoss


class iPTMLoss(BaseLoss):
    """iPTMLoss."""
    
    def __init__(
        self,
        boltz_model,
        weight: float = 0.5,
        mode: str = 'maximize',
        name: str = "iPTM"
    ):
        super().__init__(weight=weight, name=name)
        self.boltz_model = boltz_model
        self.mode = mode
        
        
        
        if hasattr(boltz_model, 'confidence_module'):
            self.confidence = boltz_model.confidence_module
        elif hasattr(boltz_model, 'confidence_head'):
            self.confidence = boltz_model.confidence_head
        elif hasattr(boltz_model, 'confidence'):
            self.confidence = boltz_model.confidence
        else:
            raise ValueError("Boltz2 model must have confidence module/head for iPTM prediction")
        
        
        if hasattr(boltz_model, 'distogram_module'):
            self.distogram = boltz_model.distogram_module
        elif hasattr(boltz_model, 'distogram'):
            self.distogram = boltz_model.distogram
        else:
            self.distogram = None
            print("[WARN] iPTMLoss: No distogram module found, using zeros")
        
        
        if hasattr(boltz_model, 'input_embedder'):
            self.input_embedder = boltz_model.input_embedder
        else:
            self.input_embedder = None
    
    def forward(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        coords: torch.Tensor,
        feats: Dict,
        metadata: Dict,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        
        num_binder = metadata.get('binder_length', 20)
        num_target = metadata.get('target_length', None)
        
        if num_target is None:
            raise ValueError("iPTM Loss requires complex mode with binder + target")
        
        
        required_fields = ['frames_idx', 'asym_id', 'token_pad_mask']
        for field in required_fields:
            if field not in feats:
                raise ValueError(f"iPTM Loss requires '{field}' in feats")
        
        try:
            
            if self.distogram is not None:
                pred_distogram_logits = self.distogram(z_trunk)
                
                if pred_distogram_logits.dim() == 5:
                    pred_distogram_logits = pred_distogram_logits[:, :, :, 0, :]  # [B, N, N, num_bins]
                elif pred_distogram_logits.dim() == 4 and pred_distogram_logits.shape[-1] < 20:
                    
                    pred_distogram_logits = pred_distogram_logits[:, :, :, 0] if pred_distogram_logits.shape[-1] > 1 else pred_distogram_logits.squeeze(-1)
            else:
                
                batch_size, num_tokens = z_trunk.shape[0], z_trunk.shape[1]
                pred_distogram_logits = torch.zeros(
                    batch_size, num_tokens, num_tokens, 64,
                    device=z_trunk.device, dtype=z_trunk.dtype
                )
            
            
            if self.input_embedder is not None:
                s_inputs = self.input_embedder(feats)
            else:
                s_inputs = s_trunk
            
            
            if not hasattr(self, '_msa_checked'):
                if 'msa' in feats:
                    print(f" [iPTMLoss] MSA found in feats: shape={feats['msa'].shape}")
                else:
                    print(' [iPTMLoss] WARNING: MSA not found in feats!')
                    if hasattr(self.confidence, 'imitate_trunk'):
                        if self.confidence.imitate_trunk:
                            print(' confidence_module.imitate_trunk=True but no MSA!')
                self._msa_checked = True
            
            
            
            confidence_output = self.confidence(
                s_inputs=s_inputs,
                s=s_trunk,
                z=z_trunk,
                x_pred=coords,
                feats=feats,  
                pred_distogram_logits=pred_distogram_logits,
                multiplicity=1
            )
            
            
            if 'iptm' in confidence_output:
                iptm = confidence_output['iptm']
            else:
                
                if 'ptm' in confidence_output:
                    print("[WARN] iPTMLoss: Using ptm as fallback for iptm")
                    iptm = confidence_output['ptm']
                else:
                    raise ValueError("Confidence output missing 'iptm' and 'ptm'")
            
            
            if self.mode == 'maximize':
                
                loss_raw = -iptm
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            
            loss = self.weight * loss_raw
            
            
            info = {
                'iptm': iptm.item() if isinstance(iptm, torch.Tensor) else iptm,
                'ptm': confidence_output.get('ptm', 0.0).item() if 'ptm' in confidence_output else 0.0,
                'loss_raw': loss_raw.item() if isinstance(loss_raw, torch.Tensor) else loss_raw,
                'loss_weighted': loss.item() if isinstance(loss, torch.Tensor) else loss,
            }
            
        except Exception as e:
            
            print(f"[ERROR] iPTMLoss computation failed: {e}")
            import traceback
            traceback.print_exc()
            
            loss = torch.tensor(0.0, device=z_trunk.device, requires_grad=True)
            info = {
                'iptm': 0.0,
                'ptm': 0.0,
                'loss_raw': 0.0,
                'loss_weighted': 0.0,
                'error': str(e)
            }
        
        return loss, info


class HotspotResidueLoss(BaseLoss):
    """HotspotResidueLoss."""
    
    def __init__(
        self,
        hotspot_indices: List[int],
        target_distance: float = 8.0,  
        weight: float = 1.0,
        beta: float = 2.0,
        name: str = "Hotspot"
    ):
        super().__init__(weight=weight, name=name)
        self.hotspot_indices = hotspot_indices
        self.target_distance = target_distance
        self.beta = beta
        
        if not hotspot_indices:
            raise ValueError("hotspot_indices cannot be empty")
    
    def forward(
        self,
        coords: torch.Tensor,
        feats: Dict,
        metadata: Dict,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        from .utils import extract_ca_atoms_by_chain
        
        
        num_binder = metadata.get('binder_length', 20)
        num_target = metadata.get('target_length', None)
        
        if num_target is None:
            raise ValueError("Hotspot Loss requires complex mode with binder + target")
        
        
        if not hasattr(self, '_verified'):
            print(f"\n [HotspotResidueLoss] info:")
            print(f"   binder_length: {num_binder}")
            print(f"   target_length (receptor): {num_target}")
            print(f" hotspot_indices (receptorchainindex): {self.hotspot_indices}")
            if 'binder_slice' in metadata:
                print(f"   binder_slice: {metadata['binder_slice']}")
            if 'receptor_slice' in metadata:
                print(f"   receptor_slice: {metadata['receptor_slice']}")
            print(f" coordinates: {coords.shape}")
            print(f" use chain_id=0 extract binder, chain_id=1 extract receptor")
            self._verified = True
        
        
        binder_ca = extract_ca_atoms_by_chain(coords, feats, chain_id=0, length=num_binder, metadata=metadata)  # [1, num_binder, 3]
        target_ca = extract_ca_atoms_by_chain(coords, feats, chain_id=1, length=num_target, metadata=metadata)  # [1, num_target, 3]
        
        
        if not hasattr(self, '_ca_verified'):
            print(f"\n [HotspotResidueLoss] CAatomextract:")
            print(f" binder_ca: {binder_ca.shape} (: [1, {num_binder}, 3])")
            print(f" target_ca: {target_ca.shape} (: [1, {num_target}, 3])")
            if target_ca.shape[1] != num_target:
                print(f" WARNING: target_calength({target_ca.shape[1]}) != num_target({num_target})")
            
            max_hotspot_idx = max(self.hotspot_indices) if self.hotspot_indices else -1
            if max_hotspot_idx >= num_target:
                print(f" ERROR: hotspotindex {max_hotspot_idx} >= receptorlength {num_target}")
            else:
                print(f" hotspotindex: [0, {max_hotspot_idx}] < {num_target}")
            self._ca_verified = True
        
        
        
        hotspot_ca = target_ca[0, self.hotspot_indices, :]  # [num_hotspots, 3]
        binder_ca_squeezed = binder_ca.squeeze(0)  # [num_binder, 3]
        
        
        # dist[i,j] = ||hotspot_ca[i] - binder_ca[j]||
        diff = hotspot_ca.unsqueeze(1) - binder_ca_squeezed.unsqueeze(0)  # [num_hotspots, num_binder, 3]
        distances = torch.norm(diff, dim=-1)  # [num_hotspots, num_binder]
        
        
        min_distances, min_indices = distances.min(dim=1)  # [num_hotspots]
        
        
        if not hasattr(self, '_dist_verified'):
            print(f"\n [HotspotResidueLoss] distancecompute:")
            print(f" distance: {distances.shape} (: [{len(self.hotspot_indices)}, {num_binder}])")
            print(f" eachhotspotdistance: {min_distances.detach().cpu().tolist()}")
            print(f" eachhotspotbinderresidueindex: {min_indices.detach().cpu().tolist()}")
            print(f" distance: {self.target_distance} Å")
            print(f" distance: {min_distances.mean().item():.2f} Å")
            self._dist_verified = True
        
        
        if not hasattr(self, '_clash_debug_step'):
            self._clash_debug_step = 0
        self._clash_debug_step += 1
        
        if self._clash_debug_step == 1 or self._clash_debug_step % 10 == 0:
            with torch.no_grad():
                clash_threshold = 4.0  
                num_clashes = (min_distances < clash_threshold).sum().item()
                min_dist = min_distances.min().item()
                print(f"\n[DEBUG] HotspotResidueLoss Clash Check (Step {self._clash_debug_step}):")
                print(f"  Min distance: {min_dist:.2f} Å (target: {self.target_distance:.2f} Å)")
                print(f"  Num hotspots with distance < {clash_threshold} Å: {num_clashes}/{len(self.hotspot_indices)}")
                if min_dist < clash_threshold:
                    print(f" WARNING: Clash detected! Distance {min_dist:.2f} Å < {clash_threshold} Å")
                    print(f"     This may cause 'collision-bounce' oscillation!")
        
        
        
        
        
        
        
        
        buffer = 2.0  
        
        target = self.target_distance  
        
        
        
        repulsion_scale = 10.0  
        repulsion_beta = 5.0    
        
        
        attraction_losses = []
        repulsion_losses = []
        
        for min_dist in min_distances:
            
            # 
            
            
            
            
            
            
            lower_bound = 2.0  
            upper_bound = 8.0  
            
            if min_dist < lower_bound:
                
                repulsion = F.softplus(lower_bound - min_dist, beta=repulsion_beta) * repulsion_scale
                attraction = torch.tensor(0.0, device=min_dist.device)
            elif min_dist <= upper_bound:
                
                
                center = (lower_bound + upper_bound) / 2  # 5Å
                attraction = 0.02 * (min_dist - center) ** 2  
                repulsion = torch.tensor(0.0, device=min_dist.device)
            else:
                
                
                deviation = min_dist - upper_bound
                if deviation < 2.0:
                    attraction = 0.5 * deviation ** 2
                else:
                    attraction = 2.0 * deviation - 2.0
                repulsion = torch.tensor(0.0, device=min_dist.device)
            
            attraction_losses.append(attraction)
            repulsion_losses.append(repulsion)
        
        
        attraction_tensor = torch.stack(attraction_losses)  # [num_hotspots]
        repulsion_tensor = torch.stack(repulsion_losses)  # [num_hotspots]
        
        
        loss_per_hotspot = attraction_tensor + repulsion_tensor
        loss_raw = loss_per_hotspot.mean()
        loss = self.weight * loss_raw
        
        
        info = {
            'min_distances': min_distances.detach().cpu().tolist(),
            'mean_min_distance': min_distances.mean().item(),
            'min_min_distance': min_distances.min().item(),  
            'target_distance': self.target_distance,
            'loss_raw': loss_raw.item(),
            'loss_weighted': loss.item(),
            'hotspot_indices': self.hotspot_indices,
            'closest_target_indices': min_indices.detach().cpu().tolist(),
        }
        
        return loss, info


class InterfaceContactLoss(BaseLoss):
    """InterfaceContactLoss."""
    
    def __init__(
        self,
        contact_threshold: float = 8.0,
        weight: float = 1.0,
        name: str = "InterfaceContact"
    ):
        super().__init__(weight=weight, name=name)
        self.contact_threshold = contact_threshold
    
    def forward(
        self,
        coords: torch.Tensor,
        feats: Dict,
        metadata: Dict,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        from .utils import extract_ca_atoms_by_chain
        
        
        num_binder = metadata.get('binder_length', 20)
        num_target = metadata.get('target_length', None)
        
        if num_target is None:
            raise ValueError("Interface Contact Loss requires complex mode")
        
        
        binder_ca = extract_ca_atoms_by_chain(coords, feats, chain_id=0, length=num_binder, metadata=metadata)  # [1, num_binder, 3]
        target_ca = extract_ca_atoms_by_chain(coords, feats, chain_id=1, length=num_target, metadata=metadata)  # [1, num_target, 3]
        
        
        binder_ca_squeezed = binder_ca.squeeze(0)
        target_ca_squeezed = target_ca.squeeze(0)
        
        
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1
        
        if self._debug_step_count == 1 or self._debug_step_count % 10 == 0:
            print(f"\n[DEBUG] InterfaceContactLoss Diagnostics (Step {self._debug_step_count}):")
            print(f"  Binder CA shape: {binder_ca_squeezed.shape}")
            print(f"  Binder CA mean: {binder_ca_squeezed.mean(dim=0).detach().cpu().numpy()}")
            print(f"  Binder CA range: [{binder_ca_squeezed.min().item():.2f}, {binder_ca_squeezed.max().item():.2f}]")
            print(f"  Target CA shape: {target_ca_squeezed.shape}")
            print(f"  Target CA mean: {target_ca_squeezed.mean(dim=0).detach().cpu().numpy()}")
            print(f"  Target CA range: [{target_ca_squeezed.min().item():.2f}, {target_ca_squeezed.max().item():.2f}]")
            
            
            with torch.no_grad():
                binder_center = binder_ca_squeezed.mean(dim=0)
                target_center = target_ca_squeezed.mean(dim=0)
                center_dist = torch.norm(binder_center - target_center).item()
                print(f"  Center-to-Center Distance: {center_dist:.2f} Å")
            
            
            if torch.isnan(binder_ca_squeezed).any() or torch.isnan(target_ca_squeezed).any():
                print(f" WARNING: NaN detected in CA coordinates!")
            if torch.isinf(binder_ca_squeezed).any() or torch.isinf(target_ca_squeezed).any():
                print(f" WARNING: Inf detected in CA coordinates!")
        
        diff = binder_ca_squeezed.unsqueeze(1) - target_ca_squeezed.unsqueeze(0)  # [num_binder, num_target, 3]
        distances = torch.norm(diff, dim=-1)  # [num_binder, num_target]
        
        
        if self._debug_step_count == 1 or self._debug_step_count % 10 == 0:
            with torch.no_grad():
                min_dist = distances.min().item()
                max_dist = distances.max().item()
                mean_dist = distances.mean().item()
                print(f"  Distance stats: min={min_dist:.2f} Å, max={max_dist:.2f} Å, mean={mean_dist:.2f} Å")
                print(f"  Contact threshold: {self.contact_threshold} Å")
        
        
        contacts = (distances < self.contact_threshold).float()
        num_contacts = contacts.sum()
        
        
        if self._debug_step_count == 1 or self._debug_step_count % 10 == 0:
            print(f"  Num contacts: {num_contacts.item()}")
            print(f"  Contact density: {(num_contacts / (num_binder * num_target)).item():.6f}")
        
        
        
        
        loss_raw = 1.0 / (num_contacts + 1.0)
        loss = self.weight * loss_raw
        
        
        contact_density = (num_contacts / (num_binder * num_target)).item()
        info = {
            'num_contacts': num_contacts.item(),
            'contact_threshold': self.contact_threshold,
            'contact_density': contact_density,
            'mean_contact_prob': contact_density,  
            'loss_raw': loss_raw.item(),
            'loss_weighted': loss.item(),
        }
        
        return loss, info


class DistogramPenaltyLoss(BaseLoss):
    """DistogramPenaltyLoss."""
    
    def __init__(
        self,
        boltz_model,
        penalty_regions: list,
        weight: float = 1.0,
        contact_threshold: float = 8.0,
        name: str = "DistogramPenalty"
    ):
        super().__init__(weight=weight, name=name)
        self.boltz_model = boltz_model
        self.penalty_regions = penalty_regions
        self.contact_threshold = contact_threshold
        
        
        
        
        self.contact_bin_end = 20  
    
    def _create_penalty_mask(
        self, 
        L: int, 
        penalty_regions: list,
        metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """ create penalty mask."""
        penalty_mask = torch.zeros(1, L, L, dtype=torch.bool)
        
        for region1, region2 in penalty_regions:
            
            if isinstance(region1, str):
                if region1 == "binder_framework":
                    
                    binder_slice = metadata.get('binder_slice')
                    optimizable_regions = metadata.get('optimizable_regions', [])
                    if binder_slice and optimizable_regions:
                        # binder_framework = binder - optimizable_regions
                        binder_mask = torch.zeros(L, dtype=torch.bool)
                        binder_mask[binder_slice] = True
                        optimizable_mask = torch.zeros(L, dtype=torch.bool)
                        for opt_start, opt_end in optimizable_regions:
                            optimizable_mask[opt_start:opt_end] = True
                        region1_mask = binder_mask & ~optimizable_mask
                    else:
                        raise ValueError("Cannot resolve 'binder_framework' without metadata")
                elif region1 == "binder":
                    
                    binder_slice = metadata.get('binder_slice')
                    if binder_slice:
                        region1_mask = torch.zeros(L, dtype=torch.bool)
                        region1_mask[binder_slice] = True
                    else:
                        raise ValueError("Cannot resolve 'binder' without binder_slice in metadata")
                elif region1 == "receptor":
                    receptor_slice = metadata.get('receptor_slice', slice(0, 0))
                    region1_mask = torch.zeros(L, dtype=torch.bool)
                    region1_mask[receptor_slice] = True
                else:
                    raise ValueError(f"Unknown region name: {region1}")
            elif isinstance(region1, slice):
                region1_mask = torch.zeros(L, dtype=torch.bool)
                region1_mask[region1] = True
            else:
                raise ValueError(f"Unsupported region type: {type(region1)}")
            
            
            if isinstance(region2, str):
                if region2 == "binder_framework":
                    binder_slice = metadata.get('binder_slice')
                    optimizable_regions = metadata.get('optimizable_regions', [])
                    if binder_slice and optimizable_regions:
                        binder_mask = torch.zeros(L, dtype=torch.bool)
                        binder_mask[binder_slice] = True
                        optimizable_mask = torch.zeros(L, dtype=torch.bool)
                        for opt_start, opt_end in optimizable_regions:
                            optimizable_mask[opt_start:opt_end] = True
                        region2_mask = binder_mask & ~optimizable_mask
                    else:
                        raise ValueError("Cannot resolve 'binder_framework' without metadata")
                elif region2 == "binder":
                    
                    binder_slice = metadata.get('binder_slice')
                    if binder_slice:
                        region2_mask = torch.zeros(L, dtype=torch.bool)
                        region2_mask[binder_slice] = True
                    else:
                        raise ValueError("Cannot resolve 'binder' without binder_slice in metadata")
                elif region2 == "receptor":
                    receptor_slice = metadata.get('receptor_slice', slice(0, 0))
                    region2_mask = torch.zeros(L, dtype=torch.bool)
                    region2_mask[receptor_slice] = True
                else:
                    raise ValueError(f"Unknown region name: {region2}")
            elif isinstance(region2, slice):
                region2_mask = torch.zeros(L, dtype=torch.bool)
                region2_mask[region2] = True
            else:
                raise ValueError(f"Unsupported region type: {type(region2)}")
            
            
            penalty_mask[0] = penalty_mask[0] | (
                region1_mask.unsqueeze(1) & region2_mask.unsqueeze(0)
            )
            
            penalty_mask[0] = penalty_mask[0] | (
                region2_mask.unsqueeze(1) & region1_mask.unsqueeze(0)
            )
        
        return penalty_mask
    
    def forward(
        self,
        z: torch.Tensor,  
        metadata: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        L = z.shape[1]
        device = z.device
        
        
        pdistogram = self.boltz_model.distogram_module(z)  # [1, L, L, 1, 64]
        pdistogram = pdistogram.squeeze(3)  # [1, L, L, 64]
        
        
        prob_distogram = F.softmax(pdistogram, dim=-1)  # [1, L, L, 64]
        
        
        contacts = torch.zeros(1, 1, 1, 64, dtype=prob_distogram.dtype, device=device)
        contacts[:, :, :, :self.contact_bin_end] = 1.0  
        prob_contact = (prob_distogram * contacts).sum(-1)  # [1, L, L]
        
        
        penalty_mask = self._create_penalty_mask(L, self.penalty_regions, metadata)
        penalty_mask = penalty_mask.to(device)
        
        
        
        prob_contact_2d = prob_contact[0]  # [L, L]
        penalty_mask_2d = penalty_mask[0]  # [L, L]
        
        penalty_contact_prob = (prob_contact_2d * penalty_mask_2d).sum()
        num_penalty_pairs = penalty_mask_2d.sum().float()
        
        
        if num_penalty_pairs > 0:
            
            avg_penalty_prob = penalty_contact_prob / num_penalty_pairs
            
            
            penalty_contact_values = prob_contact_2d[penalty_mask_2d]  # [num_penalty_pairs]
            max_penalty_prob = penalty_contact_values.max().item()
            min_penalty_prob = penalty_contact_values.min().item()
            median_penalty_prob = penalty_contact_values.median().item()
            std_penalty_prob = penalty_contact_values.std().item()
            
            
            
            high_contact_threshold = 0.4  # >40%
            high_contact_mask = penalty_contact_values > high_contact_threshold  # >40%
            medium_contact_mask = (penalty_contact_values > 0.3) & (penalty_contact_values <= high_contact_threshold)  # 30-40%
            low_contact_mask = penalty_contact_values <= 0.3  # <=30%
            
            high_contact = high_contact_mask.sum().item()
            medium_contact = medium_contact_mask.sum().item()
            low_contact = low_contact_mask.sum().item()
            
            
            
            
            if high_contact > 0:
                
                high_contact_probs = penalty_contact_values[high_contact_mask]  # [high_contact]
                
                loss_raw = high_contact_probs.mean()
            else:
                
                loss_raw = torch.tensor(0.0, device=device)
        else:
            loss_raw = torch.tensor(0.0, device=device)
            avg_penalty_prob = torch.tensor(0.0, device=device)
            max_penalty_prob = 0.0
            min_penalty_prob = 0.0
            median_penalty_prob = 0.0
            std_penalty_prob = 0.0
            high_contact = 0
            medium_contact = 0
            low_contact = 0
        
        loss = self.weight * loss_raw
        
        info = {
            'loss_raw': loss_raw.item(),
            'loss_weighted': loss.item(),
            'avg_penalty_prob': avg_penalty_prob.item(),
            'num_penalty_pairs': num_penalty_pairs.item(),
            
            'max_penalty_prob': max_penalty_prob,
            'min_penalty_prob': min_penalty_prob,
            'median_penalty_prob': median_penalty_prob,
            'std_penalty_prob': std_penalty_prob,
            'high_contact_count': high_contact,  # >40%
            'medium_contact_count': medium_contact,  # 30-40%
            'low_contact_count': low_contact,  # <=30%
        }
        
        return loss, info

