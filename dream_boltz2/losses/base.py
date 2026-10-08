"""Weighted loss base class."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """BaseLoss."""
    
    def __init__(self, weight: float = 1.0, name: str = "base_loss"):
        """  init  ."""
        super().__init__()
        self.weight = weight
        self.name = name
    
    def forward(
        self,
        z: Optional[torch.Tensor] = None,      
        coords: Optional[torch.Tensor] = None,  
        feats: Optional[Dict] = None,           
        metadata: Optional[Dict] = None,        
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        raise NotImplementedError('required forward ')
    
    def __repr__(self):
        return f"{self.__class__.__name__}(weight={self.weight}, name={self.name})"


class CompositeLoss(BaseLoss):
    """CompositeLoss."""
    
    def __init__(self, losses: list, name: str = "composite_loss"):
        """  init  ."""
        super().__init__(weight=1.0, name=name)
        self.losses = nn.ModuleList(losses)
    
    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        feats: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """forward."""
        total_loss = torch.tensor(0.0, device=self._get_device(z, coords), dtype=torch.float32)
        combined_info = {}
        
        for loss_fn in self.losses:
            loss_val, loss_info = loss_fn(
                z=z, coords=coords, feats=feats, metadata=metadata, **kwargs
            )
            
            
            total_loss = total_loss + loss_val
            
            
            for key, value in loss_info.items():
                prefixed_key = f"{loss_fn.name}_{key}"
                combined_info[prefixed_key] = value
        
        
        combined_info['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, combined_info
    
    def _get_device(self, z: Optional[torch.Tensor], coords: Optional[torch.Tensor]) -> str:
        """ get device."""
        if z is not None:
            return z.device
        elif coords is not None:
            return coords.device
        else:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __repr__(self):
        loss_names = [loss.name for loss in self.losses]
        return f"CompositeLoss(losses=[{', '.join(loss_names)}])"


# =====================================================

# =====================================================

def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """safe divide."""
    return numerator / (denominator + eps)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """safe sqrt."""
    return torch.sqrt(x + eps)


def clamp_loss(loss: torch.Tensor, max_value: float = 100.0) -> torch.Tensor:
    """clamp loss."""
    return torch.clamp(loss, max=max_value)

