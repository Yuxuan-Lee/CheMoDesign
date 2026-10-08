"""Boltz-2 diffusion wrapper used to denoise coordinates from single and pair representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import warnings


# ============================================================================

# ============================================================================

DIFFUSION_CONFIG = {
    
    'sigma_min': 0.0004,       
    'sigma_max': 160.0,        
    'sigma_data': 16.0,        
    'rho': 7.0,                
    
    
    'num_steps': 15,           
    'max_steps': 15,           
    
    
    'init_mode': 'boltz2',     
    'initial_noise_scale': 1.0,  
    
    
    'gamma_0': 0.8,            
    'gamma_min': 1.0,          
    'step_scale': 1.5,         
    'noise_scale': 1.003,      
    
    
    'use_checkpointing': False,  # Phase 1: False
                                 
}


class DiffusionWrapper(nn.Module):
    """DiffusionWrapper."""
    
    def __init__(
        self,
        boltz_model,
        num_steps: int = None,
        sigma_min: float = None,
        sigma_max: float = None,
        sigma_data: float = None,
        rho: float = None,
        gamma_0: float = None,
        gamma_min: float = None,
        step_scale: float = None,
        noise_scale: float = None,
        use_checkpointing: bool = None,
        device: str = 'cuda'
    ):
        """  init  ."""
        super().__init__()
        
        
        self.boltz_model = boltz_model
        
        
        
        if hasattr(boltz_model, 'structure_module'):
            self.diffusion = boltz_model.structure_module
        elif hasattr(boltz_model, 'diffusion_module'):
            self.diffusion = boltz_model.diffusion_module
        elif hasattr(boltz_model, 'diffusion'):
            self.diffusion = boltz_model.diffusion
        else:
            raise AttributeError(
                "Boltz2 model must have 'structure_module', 'diffusion_module', or 'diffusion' attribute. "
                f"Available attributes: {[k for k in dir(boltz_model) if not k.startswith('_')][:20]}"
            )
        
        
        if hasattr(boltz_model, 'diffusion_conditioning'):
            self.diffusion_conditioning = boltz_model.diffusion_conditioning
        else:
            raise AttributeError(
                "Boltz2 model must have 'diffusion_conditioning' attribute. "
                f"Available attributes: {[k for k in dir(boltz_model) if not k.startswith('_')][:20]}"
            )
        
        
        if hasattr(boltz_model, 'rel_pos'):
            self.rel_pos = boltz_model.rel_pos
        else:
            raise AttributeError(
                "Boltz2 model must have 'rel_pos' attribute. "
                f"Available attributes: {[k for k in dir(boltz_model) if not k.startswith('_')][:20]}"
            )
        
        
        if hasattr(boltz_model, 'input_embedder'):
            self.input_embedder = boltz_model.input_embedder
        else:
            raise AttributeError(
                "Boltz2 model must have 'input_embedder' attribute. "
                f"Available attributes: {[k for k in dir(boltz_model) if not k.startswith('_')][:20]}"
            )
        
        
        self.num_steps = num_steps or DIFFUSION_CONFIG['num_steps']
        self.sigma_min = sigma_min or DIFFUSION_CONFIG['sigma_min']
        self.sigma_max = sigma_max or DIFFUSION_CONFIG['sigma_max']
        self.sigma_data = sigma_data or DIFFUSION_CONFIG['sigma_data']
        self.rho = rho or DIFFUSION_CONFIG['rho']
        
        self.gamma_0 = gamma_0 if gamma_0 is not None else DIFFUSION_CONFIG['gamma_0']
        self.gamma_min = gamma_min if gamma_min is not None else DIFFUSION_CONFIG['gamma_min']
        self.step_scale = step_scale if step_scale is not None else DIFFUSION_CONFIG['step_scale']
        self.noise_scale = noise_scale if noise_scale is not None else DIFFUSION_CONFIG['noise_scale']
        self.use_checkpointing = use_checkpointing if use_checkpointing is not None else DIFFUSION_CONFIG['use_checkpointing']
        self.device = device
        
        
        self.fixed_noise_pool = None
        self.noise_seed = None
        
        print(f"[DiffusionWrapper] init")
        print(f" step: {self.num_steps}")
        print(f" Sigma: [{self.sigma_min}, {self.sigma_max}]")
        print(f"  Sigma data: {self.sigma_data}")
        print(f" samplingargument: gamma_0={self.gamma_0}, gamma_min={self.gamma_min}, step_scale={self.step_scale}, noise_scale={self.noise_scale}")
        print(f"  Checkpointing: {self.use_checkpointing}")
        print(f": {self.device}")
    
    # ========================================================================
    
    # ========================================================================
    
    def generate_fixed_noise_pool(
        self,
        shape: Tuple[int, int],  # (N_atoms, 3)
        max_steps: Optional[int] = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """generate fixed noise pool."""
        max_steps = max_steps or DIFFUSION_CONFIG['max_steps']
        
        if seed is None:
            
            seed = torch.randint(0, 2**32, (1,)).item()
        
        self.noise_seed = seed
        
        
        rng_state = torch.get_rng_state()
        
        
        torch.manual_seed(seed)
        noise_pool = torch.randn(max_steps, *shape, device=self.device)
        
        
        torch.set_rng_state(rng_state)
        
        self.fixed_noise_pool = noise_pool
        
        print(f"[FixedNoise] generatefixednoise")
        print(f": {noise_pool.shape}")
        print(f": {seed}")
        print(f": mean={noise_pool.mean().item():.4f}, std={noise_pool.std().item():.4f}")
        
        return noise_pool
    
    # ========================================================================
    
    # ========================================================================
    
    def sample_schedule(self, num_steps: Optional[int] = None) -> torch.Tensor:
        """sample schedule."""
        num_steps = num_steps or self.num_steps
        inv_rho = 1.0 / self.rho
        
        
        steps = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        
        
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (num_steps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        
        
        sigmas = sigmas * self.sigma_data
        
        
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        
        return sigmas
    
    # ========================================================================
    
    # ========================================================================
    
    def forward(
        self,
        s: torch.Tensor,           
        z: torch.Tensor,           
        feats: Dict,               
        num_steps: Optional[int] = None,
        use_fixed_noise: bool = True,
        start_coords: Optional[torch.Tensor] = None,
        use_physical_guidance: bool = False,  
        start_sigma: Optional[float] = None,  
    ) -> torch.Tensor:
        """forward."""
        num_steps = num_steps or self.num_steps

        
        sigmas = self.sample_schedule(num_steps)

        
        
        
        if start_sigma is not None and start_coords is not None:
            
            keep_mask = sigmas <= start_sigma
            
            keep_indices = torch.where(keep_mask)[0]
            if len(keep_indices) < 2:
                
                keep_indices = torch.arange(len(sigmas) - 2, len(sigmas), device=sigmas.device)
            
            sigmas = sigmas[keep_indices[0]:].clone()
            sigmas[0] = start_sigma
            num_steps = len(sigmas) - 1  
            print(f" [] sigma: start_sigma={start_sigma:.2f}, "
                  f"steps={num_steps}, sigmas=[{', '.join(f'{s:.1f}' for s in sigmas)}]")

        
        s_inputs_from_feats = self.input_embedder(feats)
        
        
        network_kwargs = self._prepare_conditioning(s, z, feats, s_inputs=s_inputs_from_feats)
        
        
        if 'atom_pad_mask' not in feats:
            raise ValueError("feats must contain 'atom_pad_mask'")
        atom_mask = feats['atom_pad_mask']
        
        
        x_curr = self._initialize_coords(
            feats,
            atom_mask=atom_mask,
            start_coords=start_coords,
            use_fixed_noise=use_fixed_noise,
            initial_sigma=sigmas[0]
        )
        
        
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        
        
        x_denoised = None  
        for step_idx in range(num_steps):
            sigma_tm = sigmas[step_idx]  
            sigma_t = sigmas[step_idx + 1]  
            gamma = gammas[step_idx + 1].item()  
            sigma_tm_val = sigma_tm.item()  
            sigma_t_val = sigma_t.item()  
            
            
            t_hat = sigma_tm_val * (1 + gamma)
            
            
            x_curr_centroid = x_curr.mean(dim=1, keepdim=True)
            x_curr = x_curr - x_curr_centroid
            
            
            if x_denoised is not None:
                x_denoised = x_denoised - x_denoised.mean(dim=1, keepdim=True)
            
            
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm_val**2)
            
            
            if noise_var > 0:
                eps = torch.sqrt(torch.tensor(noise_var, device=self.device, dtype=x_curr.dtype)) * torch.randn_like(x_curr)
                x_noisy = x_curr + eps
            else:
                x_noisy = x_curr
            
            
            
            
            t_hat_tensor = torch.tensor([t_hat], device=self.device, dtype=torch.float32)
            if self.use_checkpointing:
                x_denoised = torch.utils.checkpoint.checkpoint(
                    self._denoise_step,
                    x_noisy,
                    t_hat_tensor,
                    network_kwargs,
                    use_reentrant=False
                )
            else:
                x_denoised = self._denoise_step(x_noisy, t_hat_tensor, network_kwargs)
            
            
            x_denoised = x_denoised - x_curr_centroid
            
            
            if step_idx < num_steps - 1:
                
                denoised_over_sigma = (x_noisy - x_denoised) / (t_hat + 1e-8)
                x_curr = x_noisy + self.step_scale * (sigma_t_val - t_hat) * denoised_over_sigma
            else:
                
                x_curr = x_denoised
        
        
        x_final = x_curr - x_curr.mean(dim=1, keepdim=True)
        
        
        if use_physical_guidance:
            x_final = self._apply_physical_guidance(x_final, feats)
        
        return x_final
    
    # ========================================================================
    
    # ========================================================================
    
    def single_step_sds(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        feats: Dict,
        sigma: float = 56.0,
        start_coords: Optional[torch.Tensor] = None,  
        s_inputs: Optional[torch.Tensor] = None,  
        center_coords: bool = True,  
        
        old_anchor_ca: Optional[torch.Tensor] = None,  
        old_anchor_cb: Optional[torch.Tensor] = None,  
        binder_mask: Optional[torch.Tensor] = None,  
        auto_update_anchor_threshold: float = 100.0,  
    ) -> torch.Tensor:
        """single step sds."""
        # ====================================================================
        
        # ====================================================================
        #
        
        
        
        
        
        
        #
        if 'atom_pad_mask' not in feats:
            raise ValueError("feats must contain 'atom_pad_mask'")
        
        N_atoms = feats['atom_pad_mask'].shape[1]
        
        
        
        if isinstance(sigma, torch.Tensor):
            sigma_val = sigma.item() if sigma.numel() == 1 else float(sigma)
        else:
            sigma_val = float(sigma)
        
        if start_coords is not None:
            
            
            if (sigma_val < auto_update_anchor_threshold and 
                old_anchor_ca is not None and 
                old_anchor_cb is not None):
                
                print(f"[INFO] Auto-updating anchor: sigma={sigma_val:.1f} < {auto_update_anchor_threshold:.1f}")
                if start_coords.shape[-2] != N_atoms:
                    print(f"  Atom count mismatch: old={start_coords.shape[-2]}, current={N_atoms}")
                else:
                    print(f"  Sequence may have changed, regenerating anchor with template constraint")
                
                
                new_anchor, new_ca, new_cb = self.generate_anchor_with_reused_features(
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    feats=feats,
                    ca_constraint=old_anchor_ca,
                    cb_constraint=old_anchor_cb,
                    binder_mask=binder_mask,
                    s_inputs=s_inputs,
                    num_steps=30,
                    force_threshold=1.5,
                )
                
                
                X_current = new_anchor.clone().detach()
                print(f"[INFO] New anchor generated and used (shape: {X_current.shape})")
            else:
                
                
                
                
                # 
                
                
                
                
                
                # 
                
                
                
                # 
                
                
                
                
                
                # 
                if start_coords.shape[-2] != N_atoms:
                    print(f"[WARNING] Anchor atom count mismatch: anchor={start_coords.shape[-2]}, current={N_atoms}")
                    print(f"[INFO] Non-canonical residue changed atom count. Disabling anchor.")
                    print(f"[INFO] Falling back to noisy initialization (sigma_data={self.sigma_data:.1f})")
                    
                    
                    
                    X_current = torch.randn(1, N_atoms, 3, device=self.device) * self.sigma_data
                    X_current = X_current.detach()
                else:
                    
                    X_current = start_coords.clone().detach()
        else:
            
            
            
            X_current = torch.randn(1, N_atoms, 3, device=self.device) * self.sigma_data
            X_current = X_current.detach()  
        
        
        
        if center_coords:
            
            
            X_current = X_current - X_current.mean(dim=1, keepdim=True)
            if not hasattr(self, '_center_debug_printed'):
                print("[INFO] Centering ENABLED: X_current centered to origin")
                self._center_debug_printed = True
        else:
            if not hasattr(self, '_no_center_debug_printed'):
                print("[INFO] Centering DISABLED: Using raw coordinates (experimental)")
                self._no_center_debug_printed = True
        
        # ====================================================================
        
        # ====================================================================
        
        sigma_t = torch.tensor([sigma_val], device=self.device, dtype=torch.float32)
        
        # ====================================================================
        
        # ====================================================================
        epsilon = torch.randn_like(X_current)
        X_t = X_current + sigma_t * epsilon
        
        
        X_t_centroid = X_t.mean(dim=1, keepdim=True)
        
        if center_coords:
            X_t = X_t - X_t_centroid
        
        # ====================================================================
        
        # ====================================================================
        #
        
        #   logits -> Pairformer -> (s, z) -> Diffusion -> X0_hat
        #
        
        
        #
        
        
        if s_inputs is None:
            s_inputs = self.input_embedder(feats)
        
        network_kwargs = self._prepare_conditioning(s_trunk, z_trunk, feats, s_inputs=s_inputs)
        
        # ====================================================================
        
        # ====================================================================
        #
        
        with torch.set_grad_enabled(True):
            X0_hat = self._denoise_step(X_t, sigma_t, network_kwargs)
        
        
        if not hasattr(self, '_drift_debug_step'):
            self._drift_debug_step = 0
        self._drift_debug_step += 1
        
        
        
        # 
        
        
        
        
        #
        
        
        
        
        #
        X0_hat_raw_centroid = X0_hat.mean(dim=1, keepdim=True)  
        centroid_diff = (X0_hat_raw_centroid - X_t_centroid).norm().item()
        
        if self._drift_debug_step == 1 or self._drift_debug_step % 50 == 0:
            with torch.no_grad():
                X0_hat_drift = X0_hat_raw_centroid.norm().item()
                print(f"[DEBUG] Step {self._drift_debug_step}:")
                print(f"  X0_hat centroid norm: {X0_hat_drift:.4f} Å")
                print(f"  X_t centroid norm: {X_t_centroid.norm().item():.4f} Å")
                print(f"  Centroid difference: {centroid_diff:.4f} Å")
                if centroid_diff > 1.0:
                    print(f" WARNING: Large centroid diff! Before fix this would corrupt GSD gradient!")
        
        
        if center_coords:
            
            X0_hat = X0_hat - X_t_centroid
            if self._drift_debug_step == 1:
                print(f"[INFO] Centering X0_hat with X_t centroid: same coordinate system!")
        else:
            
            if self._drift_debug_step == 1:
                print(f"[INFO] Centering DISABLED: Using raw X0_hat coordinates")
        
        # ====================================================================
        
        # ====================================================================
        #
        
        
        
        
        
        
        
        
        
        #
        return X0_hat
    
    def generate_anchor(
        self,
        s_trunk: Optional[torch.Tensor] = None,  
        z_trunk: Optional[torch.Tensor] = None,  
        feats: Optional[Dict] = None,  
        num_steps: int = 200,  
        s_inputs: Optional[torch.Tensor] = None,  
        boltz_model: Optional[torch.nn.Module] = None,  
        use_boltz2_forward: bool = True,  
        recycling_steps: int = 3,  
    ) -> torch.Tensor:
        """generate anchor."""
        # ====================================================================
        
        # ====================================================================
        if feats is None:
            raise ValueError("feats must be provided")
        
        if use_boltz2_forward:
            
            model_to_use = boltz_model if boltz_model is not None else self.boltz_model
            if model_to_use is None:
                raise ValueError("boltz_model must be provided (either in __init__ or as parameter) when use_boltz2_forward=True")
        
            
            
            
            # 
            
            
            
            
            
            use_templates = getattr(model_to_use, 'use_templates', False)
            print(f"[DEBUG] Boltz2 model use_templates: {use_templates}")
            
            
            template_keys_to_check = ['template_mask', 'template_restype', 'template_frame_rot', 'template_cb', 'template_ca']
            for key in template_keys_to_check:
                if key in feats:
                    tensor = feats[key]
                    if isinstance(tensor, torch.Tensor):
                        print(f"[DEBUG] {key} original: shape={tensor.shape}, dim={tensor.dim()}")
            
            
            
            
            
            feats_fixed = self._fix_template_features_shape(feats)
            
            
            for key in template_keys_to_check:
                if key in feats_fixed:
                    tensor = feats_fixed[key]
                    if isinstance(tensor, torch.Tensor):
                        print(f"[DEBUG] {key} fixed: shape={tensor.shape}, dim={tensor.dim()}")
            
                
                
                
                
                
                
            with torch.no_grad():
                result = model_to_use(
                    feats=feats_fixed,
                    recycling_steps=recycling_steps,
                    num_sampling_steps=num_steps,
                    diffusion_samples=1,
                    max_parallel_samples=None,
                    run_confidence_sequentially=False,
                )
                
                
                X_anchor = result["sample_atom_coords"]
                
                
                X_anchor = X_anchor.detach()
            
            return X_anchor
        else:
            
            raise ValueError(
                "use_boltz2_forward must be True. "
                "We only use the standard Boltz2.forward() method to generate anchors. "
                "Please provide boltz_model and set use_boltz2_forward=True."
            )
        
        # ====================================================================
        
        # ====================================================================
        # 
        
        
        #
        
        # network_kwargs = self._prepare_conditioning(s_trunk, z_trunk, feats, s_inputs=s_inputs)
        # 
        
        # steering_args = {
        #     'physical_guidance_update': False,
        #     'fk_steering': False,
        #     'contact_guidance_update': False
        # }
        # 
        
        # atom_mask = feats.get('atom_pad_mask')
        # if atom_mask is None:
        #     raise ValueError("feats must contain 'atom_pad_mask'")
        # 
        
        # with torch.no_grad():
        #     result = self.diffusion.sample(
        #         atom_mask=atom_mask,
        #         num_sampling_steps=num_steps,
        #         multiplicity=1,
        #         max_parallel_samples=None,
        #         steering_args=steering_args,
        #         **network_kwargs
        #     )
        # 
        
        # if isinstance(result, dict):
        #     X_anchor = result.get('sample_atom_coords', result)
        # else:
        #     X_anchor = result
        # 
        
        # X_anchor = X_anchor.detach()
        # 
        # return X_anchor
    
    def _fix_template_features_shape(self, feats: Dict) -> Dict:
        """ fix template features shape."""
        feats_fixed = feats.copy()
        
        
        template_keys = [
            'template_mask', 'template_restype', 'template_frame_rot', 
            'template_frame_t', 'template_cb', 'template_ca',
            'template_mask_cb', 'template_mask_frame', 
            'query_to_template', 'visibility_ids'
        ]
        
        for key in template_keys:
            if key not in feats:
                continue
                
            tensor = feats[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            
            original_shape = tensor.shape
            original_dim = tensor.dim()
            
            
            
            # - template_restype: [B, T, L, num_classes]
            # - template_frame_rot: [B, T, L, 3, 3]
            # - template_cb, template_ca: [B, T, L, 3]
            
            
            needs_fix = False
            
            if key == 'template_mask':
                
                if original_dim == 1:
                    # [num_tokens] -> [1, 1, num_tokens]
                    needs_fix = True
                elif original_dim == 2:
                    if original_shape[0] == 1:
                        
                        needs_fix = True
                    elif original_shape[0] != 1 and original_shape[0] > 1:
                        
                        needs_fix = True
                elif original_dim == 3:
                    if original_shape[0] != 1:
                        # [num_templates, num_tokens, ...] -> [1, num_templates, num_tokens, ...]
                        needs_fix = True
            elif key == 'template_restype':
                
                if original_dim < 4:
                    needs_fix = True
                elif original_dim == 4 and original_shape[0] != 1:
                    needs_fix = True
            elif key == 'template_frame_rot':
                
                if original_dim < 5:
                    needs_fix = True
                elif original_dim == 5 and original_shape[0] != 1:
                    needs_fix = True
            elif key in ['template_frame_t', 'template_cb', 'template_ca']:
                
                if original_dim < 4:
                    needs_fix = True
                elif original_dim == 4 and original_shape[0] != 1:
                    needs_fix = True
            elif key in ['template_mask_cb', 'template_mask_frame', 'query_to_template', 'visibility_ids']:
                
                if original_dim < 3:
                    needs_fix = True
                elif original_dim == 3 and original_shape[0] != 1:
                    needs_fix = True
            
            if needs_fix:
                
                if original_dim == 1:
                    # [num_tokens] -> [1, 1, num_tokens]
                    feats_fixed[key] = tensor.unsqueeze(0).unsqueeze(0)
                elif original_dim == 2:
                    if original_shape[0] == 1:
                        
                        feats_fixed[key] = tensor.unsqueeze(1)
                    else:
                        
                        feats_fixed[key] = tensor.unsqueeze(0)
                elif original_dim == 3:
                    if original_shape[0] == 1:
                        
                        feats_fixed[key] = tensor.unsqueeze(1)
                    else:
                        
                        feats_fixed[key] = tensor.unsqueeze(0)
                elif original_dim == 4:
                    if original_shape[0] == 1:
                        
                        feats_fixed[key] = tensor.unsqueeze(1)
                    else:
                        
                        feats_fixed[key] = tensor.unsqueeze(0)
                elif original_dim == 5:
                    if original_shape[0] != 1:
                        
                        feats_fixed[key] = tensor.unsqueeze(0)
                
                print(f"[DEBUG] Fixed {key}: {original_shape} -> {feats_fixed[key].shape}")
        
        return feats_fixed
    
    # ========================================================================
    
    # ========================================================================
    # 
    
    
    
    
    
    
    
    
    
    #
    
    
    
    
    #
    
    
    # ========================================================================
    
    def extract_ca_cb_coords(
        self,
        X_full: torch.Tensor,  # [1, N_atoms, 3]
        feats: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """extract ca cb coords."""
        
        if 'token_to_rep_atom' not in feats:
            raise ValueError("feats must contain 'token_to_rep_atom' to extract CB coordinates")
        
        token_to_rep_atom = feats['token_to_rep_atom']  
        if token_to_rep_atom.dim() == 2:
            token_to_rep_atom = token_to_rep_atom.unsqueeze(0)
        
        # X_full: [1, N_atoms, 3], token_to_rep_atom: [1, N_tokens, N_atoms]
        X_CB = torch.bmm(token_to_rep_atom.float(), X_full)  # [1, N_tokens, 3]
        
        
        if 'token_to_center_atom' in feats:
            token_to_center_atom = feats['token_to_center_atom']  
            if token_to_center_atom.dim() == 2:
                token_to_center_atom = token_to_center_atom.unsqueeze(0)
            
            X_CA = torch.bmm(token_to_center_atom.float(), X_full)  # [1, N_tokens, 3]
        else:
            
            
            print("[WARNING] token_to_center_atom not found, using token_to_rep_atom for CA")
            X_CA = X_CB.clone()
        
        return X_CA, X_CB
    
    def inject_ca_cb_template(
        self,
        feats: Dict,
        ca_coords: torch.Tensor,  
        cb_coords: torch.Tensor,  
        binder_mask: Optional[torch.Tensor] = None,  
        force_threshold: float = 1.5,  
    ) -> Dict:
        """inject ca cb template."""
        feats = dict(feats)  
        
        
        if ca_coords.dim() == 3:
            ca_coords = ca_coords.squeeze(0)  # [1, N_tokens, 3] -> [N_tokens, 3]
        if cb_coords.dim() == 3:
            cb_coords = cb_coords.squeeze(0)  # [1, N_tokens, 3] -> [N_tokens, 3]
        
        N_tokens = ca_coords.shape[0]
        device = ca_coords.device
        
        
        # 
        
        
        
        # 
        
        
        
        
        # 
        
        
        
        
        
        
        
        
        # ================================================================
        
        # ================================================================
        num_templates = 1
        
        
        template_cb = cb_coords.unsqueeze(0)  # [1, N_tokens, 3]
        template_ca = ca_coords.unsqueeze(0)  # [1, N_tokens, 3]
        
        # Mask:[num_templates, N_tokens]
        if binder_mask is not None:
            
            
            if binder_mask.dim() == 1:
                if binder_mask.shape[0] != N_tokens:
                    print(f"[WARNING] binder_mask shape mismatch: {binder_mask.shape} vs N_tokens={N_tokens}")
                    
                    if binder_mask.shape[0] < N_tokens:
                        
                        binder_mask_padded = torch.zeros(N_tokens, dtype=binder_mask.dtype, device=device)
                        binder_mask_padded[:binder_mask.shape[0]] = binder_mask
                        binder_mask = binder_mask_padded
                    else:
                        
                        binder_mask = binder_mask[:N_tokens]
                template_mask_cb = binder_mask.float().view(1, N_tokens)
            else:
                template_mask_cb = binder_mask.float()
                if template_mask_cb.dim() == 2:
                    template_mask_cb = template_mask_cb.unsqueeze(0) if template_mask_cb.shape[0] != 1 else template_mask_cb
        else:
            
            template_mask_cb = torch.ones(1, N_tokens, device=device)
        
        print(f"[DEBUG] Template mask setup:")
        print(f"  template_mask_cb shape: {template_mask_cb.shape}")
        print(f"  template_mask_cb sum: {template_mask_cb.sum().item()} / {N_tokens}")
        if template_mask_cb.shape[0] == 1:
            mask_array = template_mask_cb[0, :].cpu().numpy()
            print(f"  template_mask_cb (first 5): {mask_array[:5]}")
            print(f"  template_mask_cb (last 5): {mask_array[-5:]}")
        
        
        print(f"[DEBUG] Constraint coordinates stats:")
        print(f"  template_cb shape: {template_cb.shape}")
        if binder_mask is not None and binder_mask.sum() > 0:
            binder_indices = torch.where(binder_mask)[0]
            print(f"  template_cb (binder region) mean: {template_cb[0, binder_indices, :].mean().item():.3f}")
            print(f"  template_cb (binder region) std: {template_cb[0, binder_indices, :].std().item():.3f}")
            print(f"  template_ca (binder region) mean: {template_ca[0, binder_indices, :].mean().item():.3f}")
            print(f"  template_ca (binder region) std: {template_ca[0, binder_indices, :].std().item():.3f}")
        
        
        # frame_rot: [num_templates, N_tokens, 3, 3]
        template_frame_rot = torch.eye(3, device=device).view(1, 1, 3, 3).expand(num_templates, N_tokens, 3, 3).clone()
        # frame_t: [num_templates, N_tokens, 3]
        template_frame_t = ca_coords.unsqueeze(0)
        # frame_mask: [num_templates, N_tokens]
        template_mask_frame = template_mask_cb.clone()
        
        # Restype:[num_templates, N_tokens, num_classes]
        num_classes = 33
        template_restype = torch.zeros(num_templates, N_tokens, num_classes, device=device)
        
        # Template mask:[num_templates, N_tokens]
        template_mask = template_mask_cb.clone()
        
        # ================================================================
        
        # ================================================================
        
        template_force = torch.tensor([True], device=device)
        
        template_force_threshold = torch.tensor([force_threshold], device=device, dtype=torch.float32)
        
        # ================================================================
        
        # ================================================================
        
        visibility_ids = torch.zeros(num_templates, N_tokens, device=device)
        query_to_template = torch.zeros(num_templates, N_tokens, dtype=torch.long, device=device)
        
        # ================================================================
        
        # ================================================================
        
        has_batch_dim = False
        if 'atom_pad_mask' in feats:
            has_batch_dim = (feats['atom_pad_mask'].dim() >= 2 and feats['atom_pad_mask'].shape[0] == 1)
        
        if has_batch_dim:
            
            feats['template_cb'] = template_cb.unsqueeze(0)
            feats['template_ca'] = template_ca.unsqueeze(0)
            feats['template_mask_cb'] = template_mask_cb.unsqueeze(0)
            feats['template_frame_rot'] = template_frame_rot.unsqueeze(0)
            feats['template_frame_t'] = template_frame_t.unsqueeze(0)
            feats['template_mask_frame'] = template_mask_frame.unsqueeze(0)
            feats['template_restype'] = template_restype.unsqueeze(0)
            feats['template_mask'] = template_mask.unsqueeze(0)
            feats['visibility_ids'] = visibility_ids.unsqueeze(0)
            feats['query_to_template'] = query_to_template.unsqueeze(0)
        else:
            feats['template_cb'] = template_cb
            feats['template_ca'] = template_ca
            feats['template_mask_cb'] = template_mask_cb
            feats['template_frame_rot'] = template_frame_rot
            feats['template_frame_t'] = template_frame_t
            feats['template_mask_frame'] = template_mask_frame
            feats['template_restype'] = template_restype
            feats['template_mask'] = template_mask
            feats['visibility_ids'] = visibility_ids
            feats['query_to_template'] = query_to_template
        
        
        feats['template_force'] = template_force
        feats['template_force_threshold'] = template_force_threshold
        
        print(f"[DEBUG] Template features injected:")
        print(f"  template_cb shape: {feats['template_cb'].shape}")
        print(f"  template_mask_cb shape: {feats['template_mask_cb'].shape}")
        print(f"  template_force: {feats['template_force']}")
        print(f"  template_force_threshold: {feats['template_force_threshold']} Å")
        
        return feats
    
    def generate_anchor_with_reused_features(
        self,
        s_trunk: torch.Tensor,  
        z_trunk: torch.Tensor,  
        feats: Dict,
        ca_constraint: Optional[torch.Tensor] = None,  
        cb_constraint: Optional[torch.Tensor] = None,  
        binder_mask: Optional[torch.Tensor] = None,  # [N_tokens] bool
        s_inputs: Optional[torch.Tensor] = None,  
        num_steps: int = 30,  
        force_threshold: float = 1.5,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """generate anchor with reused features."""
        
        if ca_constraint is not None and cb_constraint is not None:
            print(f"[DEBUG] Generating anchor with Template constraint (reusing s_trunk, z_trunk)")
            print(f"  CA constraint shape: {ca_constraint.shape}")
            print(f"  CB constraint shape: {cb_constraint.shape}")
            print(f"  Force threshold: {force_threshold} Å")
            feats = self.inject_ca_cb_template(
                feats, ca_constraint, cb_constraint, binder_mask, force_threshold
            )
        else:
            print(f"[WARNING] Generating anchor WITHOUT Template constraint (first anchor)")
        
        
        feats = self._fix_template_features_shape(feats)
        
        
        
        if s_inputs is None:
            s_inputs = self.input_embedder(feats)
        
        network_kwargs = self._prepare_conditioning(s_trunk, z_trunk, feats, s_inputs=s_inputs)
        
        
        sigmas = self.sample_schedule(num_steps)
        
        
        if 'atom_pad_mask' not in feats:
            raise ValueError("feats must contain 'atom_pad_mask'")
        atom_mask = feats['atom_pad_mask']
        
        x_curr = self._initialize_coords(
            feats,
            atom_mask=atom_mask,
            start_coords=None,  
            use_fixed_noise=True,
            initial_sigma=sigmas[0]
        )
        
        
        
        with torch.no_grad():  
            X_anchor = self.forward(
                s=s_trunk,
                z=z_trunk,
                feats=feats,
                num_steps=num_steps,
                use_fixed_noise=True,
                start_coords=None
            )
            X_anchor = X_anchor.detach()
            
            
            X_CA, X_CB = self.extract_ca_cb_coords(X_anchor, feats)
            X_CA = X_CA.detach()
            X_CB = X_CB.detach()
        
        print(f"[DEBUG] Anchor generated (reused features):")
        print(f"  Full coords shape: {X_anchor.shape}")
        print(f"  CA coords shape: {X_CA.shape}")
        print(f"  CB coords shape: {X_CB.shape}")
        
        return X_anchor, X_CA, X_CB
    
    def generate_anchor_with_template_constraint(
        self,
        feats: Dict,
        ca_constraint: Optional[torch.Tensor] = None,  
        cb_constraint: Optional[torch.Tensor] = None,  
        binder_mask: Optional[torch.Tensor] = None,  # [N_tokens] bool
        boltz_model: Optional[torch.nn.Module] = None,
        num_steps: int = 30,  
        recycling_steps: int = 0,  
        force_threshold: float = 1.5,  
        
        s_trunk: Optional[torch.Tensor] = None,  
        z_trunk: Optional[torch.Tensor] = None,  
        s_inputs: Optional[torch.Tensor] = None,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """generate anchor with template constraint."""
        
        if s_trunk is not None and z_trunk is not None:
            return self.generate_anchor_with_reused_features(
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                feats=feats,
                ca_constraint=ca_constraint,
                cb_constraint=cb_constraint,
                binder_mask=binder_mask,
                s_inputs=s_inputs,
                num_steps=num_steps,
                force_threshold=force_threshold,
            )
        
        
        model_to_use = boltz_model if boltz_model is not None else self.boltz_model
        if model_to_use is None:
            raise ValueError("boltz_model must be provided when s_trunk/z_trunk are not provided")
        
        
        if ca_constraint is not None and cb_constraint is not None:
            print(f"[DEBUG] Generating anchor with Template constraint (full forward)")
            print(f"  CA constraint shape: {ca_constraint.shape}")
            print(f"  CB constraint shape: {cb_constraint.shape}")
            print(f"  Force threshold: {force_threshold} Å")
            feats = self.inject_ca_cb_template(
                feats, ca_constraint, cb_constraint, binder_mask, force_threshold
            )
        else:
            
            print(f"[WARNING] Generating anchor WITHOUT Template constraint (first anchor)")
        
        
        feats = self._fix_template_features_shape(feats)
        
        
        with torch.no_grad():
            result = model_to_use(
                feats=feats,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_steps,
                diffusion_samples=1,
                max_parallel_samples=None,
                run_confidence_sequentially=False,
            )
            
            
            X_anchor = result["sample_atom_coords"].detach()
            
            
            X_CA, X_CB = self.extract_ca_cb_coords(X_anchor, feats)
            X_CA = X_CA.detach()
            X_CB = X_CB.detach()
        
        print(f"[DEBUG] Anchor generated (full forward):")
        print(f"  Full coords shape: {X_anchor.shape}")
        print(f"  CA coords shape: {X_CA.shape}")
        print(f"  CB coords shape: {X_CB.shape}")
        
        return X_anchor, X_CA, X_CB
    
    def _compute_sigma_at_time(self, t: float) -> torch.Tensor:
        """ compute sigma at time."""
        
        sigma_schedule = self.sample_schedule(num_steps=1000)
        
        
        idx = int(t * 999)
        idx = max(0, min(999, idx))  
        
        return sigma_schedule[idx]
    
    # ========================================================================
    
    # ========================================================================
    
    def _prepare_conditioning(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        feats: Dict,
        s_inputs: Optional[torch.Tensor] = None  
    ) -> Dict:
        """ prepare conditioning."""
        
        relative_position_encoding = self.rel_pos(feats)
        
        
        
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = self.diffusion_conditioning(
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            feats=feats
        )
        
        
        diffusion_conditioning = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }
        
        
        
        
        
        if s_inputs is None:
            raise ValueError(
                "s_inputs must be provided! It must be generated from the current feats using input_embedder(feats). "
                "Using s_trunk as a fallback is incorrect because s_inputs and feats must match."
            )
        
        
        if not hasattr(self, '_s_inputs_warning_printed'):
            if torch.allclose(s_inputs, s_trunk, rtol=1e-3, atol=1e-3):
                print('[WARNING] s_inputs and s_trunk ,this(ifsequencenot)')
            else:
                print('[INFO] s_inputs and s_trunk not,this！')
            self._s_inputs_warning_printed = True
        
        network_kwargs = {
            's_inputs': s_inputs,  
            's_trunk': s_trunk,   
            'feats': feats,       
            'diffusion_conditioning': diffusion_conditioning,  
            
        }
        
        return network_kwargs
    
    def _denoise_step(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        network_kwargs: Dict
    ) -> torch.Tensor:
        """ denoise step."""
        
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)  # [1]
        
        
        
        denoised_coords = self.diffusion.preconditioned_network_forward(
            noised_atom_coords=x_noisy,
            sigma=sigma,
            network_condition_kwargs=network_kwargs
        )
        
        
        if not hasattr(self, '_debug_printed'):
            print("\n" + "="*60)
            print("DEBUG: Diffusion Output Verification")
            print("="*60)
            print(f"Input X_noisy:")
            print(f"  Shape: {x_noisy.shape}")
            print(f"  Mean:  {x_noisy.mean():.4f}")
            print(f"  Std:   {x_noisy.std():.4f}")
            print(f"Output denoised_coords:")
            print(f"  Shape: {denoised_coords.shape}")
            print(f"  Mean:  {denoised_coords.mean():.4f}")
            print(f"  Std:   {denoised_coords.std():.4f}")
            sigma_val = sigma.item() if hasattr(sigma, 'item') else float(sigma)
            print(f"Sigma: {sigma_val:.4f}")
            print(f"Expected: Output std should be ~10-20 (protein scale)")
            if denoised_coords.std() < 5 or denoised_coords.std() > 50:
                print(f"  WARNING: Output std ({denoised_coords.std():.2f}) seems unusual!")
            print("="*60 + "\n")
            self._debug_printed = True
        
        
        return denoised_coords
    
    def _apply_physical_guidance(
        self,
        coords: torch.Tensor,  
        feats: Dict,  
        num_gd_steps: int = 20,  
    ) -> torch.Tensor:
        """ apply physical guidance."""
        try:
            
            from boltz.model.potentials.potentials import get_potentials
            
            
            steering_args = {
                "physical_guidance_update": True,
                "fk_steering": False,
                "contact_guidance_update": False,
                "num_gd_steps": num_gd_steps,
            }
            
            
            potentials = get_potentials(steering_args, boltz2=True)
            
            if len(potentials) == 0:
                print(' warning:nottopotentials,skipoptimization')
                return coords
            
            print(f" optimization({len(potentials)}potentials,{num_gd_steps}stepgradient)")
            
            
            guidance_update = torch.zeros_like(coords)
            
            
            with torch.no_grad():  
                for guidance_step in range(num_gd_steps):
                    energy_gradient = torch.zeros_like(coords)
                    
                    
                    for potential in potentials:
                        
                        steering_t = 1.0  
                        parameters = potential.compute_parameters(steering_t)
                        
                        
                        if (
                            parameters.get("guidance_weight", 0) > 0
                            and (guidance_step) % parameters.get("guidance_interval", 1) == 0
                        ):
                            
                            grad = potential.compute_gradient(
                                coords + guidance_update,
                                feats,
                                parameters,
                            )
                            
                            energy_gradient += parameters["guidance_weight"] * grad
                    
                    
                    guidance_update -= energy_gradient
                
                
                coords_optimized = coords + guidance_update
                
                
                coord_diff = (coords_optimized - coords).norm().item()
                print(f" optimization:coordinates {coord_diff:.4f} Å")
            
            return coords_optimized
            
        except Exception as e:
            print(f" optimizationfailed: {e}")
            import traceback
            traceback.print_exc()
            return coords  
    
    def _initialize_coords(
        self,
        feats: Dict,
        atom_mask: torch.Tensor,
        start_coords: Optional[torch.Tensor] = None,
        use_fixed_noise: bool = True,
        initial_sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """ initialize coords."""
        shape = (*atom_mask.shape, 3)  # [1, N_atoms, 3]
        
        
        if start_coords is not None:
            
            x_init = start_coords
        else:
            
            if use_fixed_noise and self.fixed_noise_pool is not None:
                
                noise = self.fixed_noise_pool[0:1]  # [1, N_atoms, 3]
            else:
                
                noise = torch.randn(shape, device=self.device)
            
            
            x_init = initial_sigma * noise
        
        return x_init


# ============================================================================

# ============================================================================

'\nPhase 2 (not)\n\n1. (Warm Start)- already:\n \n:\n "needanotsequence,this！"\n \n:\n - **sequence**(already binder sequence)\n - notcoordinates\n - sequencecan SequenceRepresentation init\n \n coordinates(optional,):\n ```python\n # usetimesoptimizationcoordinatesas\n coords_prev = diffusion.forward(s, z, feats)\n \n # timesoptimization(fromtimescoordinates)\n coords_next = diffusion.forward(s, z, feats, start_coords=coords_prev)\n ```\n \n note:\n - \n - vs GSD,timesfromnoise\n - Phase 1 notuse\n\n2. step(ifneed):\n \n current:fixed 15 step\n not:canoptimization\n \n ```python\n #:step\n if opt_step < 50:\n num_steps = 5\n #:stepoptimization\n elif opt_step < 100:\n num_steps = 10\n else:\n num_steps = 15\n \n coords = diffusion.forward(s, z, feats, num_steps=num_steps)\n ```\n \n note:\n - ifstep,fixednoiseneed 15 \n - N stepuse N \n\n3. Gradient checkpointing:\n \n current:default\n not:ifnot,can\n \n ```python\n diffusion = DiffusionWrapper(\n boltz_model,\n use_checkpointing=True # , ~30% compute\n )\n ```\n \n:\n - notsavein()\n - compute(compute)\n - for\n'

