"""Pairformer wrapper. Scales pair representation z by (1 - alpha) except receptor-receptor and functional-anchor pairs."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List


class PairformerWrapper(nn.Module):
    """PairformerWrapper."""
    
    def __init__(
        self,
        boltz_model,
        num_recycles: int = 3,
        freeze_recycle_layers: bool = True,
        monitor_z: bool = True,
        feature_prep: Optional = None,  
        use_checkpointing: bool = False,  
        use_interface_mask: bool = False  
    ):
        """  init  ."""
        super().__init__()
        
        self.num_recycles = num_recycles
        self.freeze_recycle_layers = freeze_recycle_layers
        self.monitor_z = monitor_z
        self.feature_prep = feature_prep  
        self.use_checkpointing = use_checkpointing  
        self.use_interface_mask = use_interface_mask  
        
        
        self.input_embedder = boltz_model.input_embedder
        from dream_boltz2.model.soft_chemistry import install_embedder_capture
        install_embedder_capture(self.input_embedder)
        self.s_init = boltz_model.s_init
        self.z_init_1 = boltz_model.z_init_1
        self.z_init_2 = boltz_model.z_init_2
        self.rel_pos = boltz_model.rel_pos
        self.token_bonds = boltz_model.token_bonds
        
        self.bond_type_feature = getattr(boltz_model, 'bond_type_feature', False)
        if self.bond_type_feature:
            self.token_bonds_type = boltz_model.token_bonds_type
        else:
            self.token_bonds_type = None
        self.contact_conditioning = boltz_model.contact_conditioning
        
        
        self.s_norm = boltz_model.s_norm
        self.z_norm = boltz_model.z_norm
        self.s_recycle = boltz_model.s_recycle
        self.z_recycle = boltz_model.z_recycle
        
        
        
        
        
        if hasattr(boltz_model.msa_module, '_orig_mod'):
            self.msa_module = boltz_model.msa_module._orig_mod
            print('[PairformerWrapper] usenot msa_module(_orig_mod)')
        else:
            self.msa_module = boltz_model.msa_module
            print('[PairformerWrapper] msa_module not,use')
        
        if hasattr(boltz_model.pairformer_module, '_orig_mod'):
            self.pairformer_module = boltz_model.pairformer_module._orig_mod
            print('[PairformerWrapper] usenot pairformer_module(_orig_mod)')
        else:
            self.pairformer_module = boltz_model.pairformer_module
            print('[PairformerWrapper] pairformer_module not,use')
        
        
        self.use_templates = getattr(boltz_model, 'use_templates', False)
        if self.use_templates and hasattr(boltz_model, 'template_module'):
            self.template_module = boltz_model.template_module
            print('[PairformerWrapper] Template already')
        else:
            self.template_module = None
            print('[PairformerWrapper] Template not(notconfig use_templates=True)')
        
        
        if freeze_recycle_layers:
            self._freeze_recycle_layers()
            print('[PairformerWrapper] Recycling alreadyfrozen( Z )')
        else:
            print('[PairformerWrapper] Recycling not frozen(not)')
        
        if use_checkpointing:
            print('[PairformerWrapper] Gradient Checkpointing already(,)')
        else:
            print('[PairformerWrapper] Gradient Checkpointing not(receptorOOM)')
        
        
        self.z_stats_history = []
    
    @staticmethod
    def identify_interface_residues_from_distogram(
        target_distogram: torch.Tensor,  # [L, L, 64]
        binder_slice: slice,
        receptor_slice: slice,
        contact_cutoff: float = 9.0,  
        min_prob: float = 0.1  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """identify interface residues from distogram."""
        L = target_distogram.shape[0]
        receptor_start = receptor_slice.start
        receptor_end = receptor_slice.stop
        binder_start = binder_slice.start
        binder_end = binder_slice.stop
        
        
        
        min_dist, max_dist = 2.0, 22.0
        num_bins = target_distogram.shape[-1]
        boundaries = torch.linspace(min_dist, max_dist, num_bins - 1, device=target_distogram.device)
        contact_bin_max = (boundaries < contact_cutoff).sum().item()  
        
        
        
        
        
        # 
        
        
        
        
        
        
        
        binder_length = binder_end - binder_start
        receptor_length = receptor_end - receptor_start
        
        
        
        
        
        
        
        
        interface_distogram_1 = target_distogram[binder_start:binder_end, receptor_start:receptor_end, :]  # [B, R, 64]
        contact_probs_1 = interface_distogram_1[:, :, :contact_bin_max].sum(dim=-1)  # [B, R]
        total_contact_1 = contact_probs_1.sum().item()
        
        
        
        
        interface_distogram_2 = target_distogram[0:binder_length, binder_length:binder_length+receptor_length, :]  # [B, R, 64]
        contact_probs_2 = interface_distogram_2[:, :, :contact_bin_max].sum(dim=-1)  # [B, R]
        total_contact_2 = contact_probs_2.sum().item()
        
        
        if total_contact_1 >= total_contact_2:
            
            
            
            
            contact_probs = contact_probs_1  # [B, R]
            use_direction_2 = False
        else:
            
            
            
            
            
            
            
            contact_probs = contact_probs_2  # [B, R]
            use_direction_2 = True
        
        
        interface_mask = contact_probs > min_prob  # [B, R] - bool tensor
        
        
        
        binder_indices_local, receptor_indices_local = torch.where(interface_mask)
        
        
        
        
        
        interface_binder_indices = binder_indices_local + binder_start
        interface_receptor_indices = receptor_indices_local + receptor_start
        
        return interface_receptor_indices, interface_binder_indices
    
    def _freeze_recycle_layers(self):
        """ freeze recycle layers."""
        
        for param in self.s_recycle.parameters():
            param.requires_grad = False
        
        
        for param in self.z_recycle.parameters():
            param.requires_grad = False
        
        
        for param in self.s_norm.parameters():
            param.requires_grad = False
        for param in self.z_norm.parameters():
            param.requires_grad = False
    
    def apply_creativity_scaling(
        self,
        z: torch.Tensor,  # [B, L, L, D]
        creativity: float,
        cdr_mask: Optional[torch.Tensor] = None,  
        receptor_mask: Optional[torch.Tensor] = None,  
        binder_mask: Optional[torch.Tensor] = None,  
        hotspot_indices: Optional[list] = None,  
        receptor_start: int = 0,  
        protected_positions: Optional[List[int]] = None,  
        ligand_mask: Optional[torch.Tensor] = None,  
        bond_constraint_pairs: Optional[torch.Tensor] = None,  
        scaling_residues: Optional[List[int]] = None,  
        scaling_residues_only: bool = False,  
    ) -> torch.Tensor:
        """apply creativity scaling."""
        if creativity <= 0.0:
            return z  
        
        
        if scaling_residues_only and scaling_residues is not None and len(scaling_residues) > 0:
            L = z.shape[1]
            device = z.device
            scale = 1.0 - creativity
            
            
            scaling_residue_mask = torch.zeros(L, dtype=torch.bool, device=device)
            for idx in scaling_residues:
                if 0 <= idx < L:
                    scaling_residue_mask[idx] = True
            
            if scaling_residue_mask.any():
                
                scaling_mask = torch.ones(L, L, device=device)  
                scaling_residue_2d = scaling_residue_mask[:, None] | scaling_residue_mask[None, :]  # [L, L]
                scaling_mask[scaling_residue_2d] = scale  
                
                if not hasattr(self, '_scaling_residues_only_debug_printed'):
                    scaling_count = scaling_residue_mask.sum().item()
                    scaled_pairs_count = scaling_residue_2d.sum().item()
                    print(f"[Creativity] scalingmodealready")
                    print(f" - scalingresidue: {scaling_count}")
                    print(f" - scalingresidueandallresiduevs({scaled_pairs_count})")
                    print(f" - allZ pairunscaled(defaultscaling)")
                    print(f" - scalingresiduelist(0-based): {scaling_residues}")
                    self._scaling_residues_only_debug_printed = True
                
                
                scaling_mask = scaling_mask.view(1, L, L, 1)
                z_scaled = z * scaling_mask
                return z_scaled
            else:
                
                if not hasattr(self, '_scaling_residues_only_warned'):
                    print(f"[Creativity] scalingmodealready,residueno,todefaultscaling")
                    self._scaling_residues_only_warned = True
        
        
        scale = 1.0 - creativity  
        
        L = z.shape[1]
        device = z.device
        
        
        hotspot_mask = torch.zeros(L, dtype=torch.bool, device=device)
        if hotspot_indices is not None and len(hotspot_indices) > 0:
            
            global_hotspot_indices = [receptor_start + idx for idx in hotspot_indices]
            for idx in global_hotspot_indices:
                if 0 <= idx < L:
                    hotspot_mask[idx] = True
        
        
        protected_mask = torch.zeros(L, dtype=torch.bool, device=device)
        if protected_positions is not None and len(protected_positions) > 0:
            for idx in protected_positions:
                if 0 <= idx < L:
                    protected_mask[idx] = True
        
        
        if ligand_mask is not None:
            if ligand_mask.dim() > 1:
                ligand_mask = ligand_mask.squeeze()
            ligand_mask = ligand_mask.bool()
        else:
            ligand_mask = torch.zeros(L, dtype=torch.bool, device=device)
        
        
        scaling_mask = torch.ones(L, L, device=device)
        
        if cdr_mask is not None:
            # ========================================
            
            # ========================================
            if cdr_mask.dim() > 1:
                cdr_mask = cdr_mask.squeeze()
            cdr_mask = cdr_mask.bool()
            
            if receptor_mask is not None:
                if receptor_mask.dim() > 1:
                    receptor_mask = receptor_mask.squeeze()
                receptor_mask = receptor_mask.bool()
            
            if binder_mask is not None:
                if binder_mask.dim() > 1:
                    binder_mask = binder_mask.squeeze()
                binder_mask = binder_mask.bool()
                framework_mask = binder_mask & (~cdr_mask)
            else:
                framework_mask = torch.zeros(L, dtype=torch.bool, device=device)
            
            
            non_hotspot_receptor = receptor_mask & (~hotspot_mask) if receptor_mask is not None else torch.zeros(L, dtype=torch.bool, device=device)
            
            
            cdr_2d = cdr_mask[:, None] & cdr_mask[None, :]
            scaling_mask[cdr_2d] = scale
            
            
            cdr_non_hotspot = cdr_mask[:, None] & non_hotspot_receptor[None, :]
            non_hotspot_cdr = non_hotspot_receptor[:, None] & cdr_mask[None, :]
            scaling_mask[cdr_non_hotspot] = scale
            scaling_mask[non_hotspot_cdr] = scale
            
            
            
            
            
            
            
            
            if receptor_mask is not None:
                framework_receptor = framework_mask[:, None] & receptor_mask[None, :]
                receptor_framework = receptor_mask[:, None] & framework_mask[None, :]
                scaling_mask[framework_receptor] = scale
                scaling_mask[receptor_framework] = scale
            
            
            
            
            
            if not hasattr(self, '_creativity_debug_printed'):
                cdr_count = cdr_mask.sum().item()
                framework_count = framework_mask.sum().item()
                receptor_count = receptor_mask.sum().item() if receptor_mask is not None else 0
                hotspot_count = hotspot_mask.sum().item()
                scaled_pairs = (scaling_mask < 1.0).sum().item()
                print(f"[Creativity] Nanobody/Fab mode()")
                print(f" - CDR residue: {cdr_count}")
                print(f" - frameworkresidue: {framework_count}")
                print(f" - Receptor residue: {receptor_count}")
                print(f" - hotspotresidue: {hotspot_count}")
                print(f" - scalingresiduevs: {scaled_pairs} / {L*L}")
                print(f" - scaling: {scale:.2f} (creativity={creativity:.2f})")
                print(f" - CDR-hotspot: unscaled(keep)")
                print(f" - framework-receptor: scaling")
                print(f" - framework-framework: unscaled(keepbackbone)")
                self._creativity_debug_printed = True
        
        elif binder_mask is not None:
            # ========================================
            
            # ========================================
            if binder_mask.dim() > 1:
                binder_mask = binder_mask.squeeze()
            binder_mask = binder_mask.bool()
            
            if receptor_mask is not None:
                if receptor_mask.dim() > 1:
                    receptor_mask = receptor_mask.squeeze()
                receptor_mask = receptor_mask.bool()
            else:
                receptor_mask = ~binder_mask
            
            
            non_hotspot_receptor = receptor_mask & (~hotspot_mask)
            
            
            binder_2d = binder_mask[:, None] & binder_mask[None, :]
            scaling_mask[binder_2d] = scale
            
            
            binder_non_hotspot = binder_mask[:, None] & non_hotspot_receptor[None, :]
            non_hotspot_binder = non_hotspot_receptor[:, None] & binder_mask[None, :]
            scaling_mask[binder_non_hotspot] = scale
            scaling_mask[non_hotspot_binder] = scale
            
            
            
            
            
            
            
            if not hasattr(self, '_creativity_debug_printed'):
                binder_count = binder_mask.sum().item()
                receptor_count = receptor_mask.sum().item()
                hotspot_count = hotspot_mask.sum().item()
                scaled_pairs = (scaling_mask < 1.0).sum().item()
                print(f"[Creativity] mode()")
                print(f" - Binder residue: {binder_count}")
                print(f" - Receptor residue: {receptor_count}")
                print(f" - hotspotresidue: {hotspot_count}")
                print(f" - scalingresiduevs: {scaled_pairs} / {L*L}")
                print(f" - scaling: {scale:.2f} (creativity={creativity:.2f})")
                if hotspot_count > 0:
                    print(f" - Binder-hotspot: unscaled(keep)")
                self._creativity_debug_printed = True
        
        else:
            
            if not hasattr(self, '_creativity_no_mask_warned'):
                print(f"[Creativity] not mask,skipalphascaling")
                self._creativity_no_mask_warned = True
            return z
        
        # ========================================
        
        # ========================================
        
        if protected_mask.any() and receptor_mask is not None:
            protected_receptor = protected_mask[:, None] & receptor_mask[None, :]
            receptor_protected = receptor_mask[:, None] & protected_mask[None, :]
            scaling_mask[protected_receptor] = 1.0
            scaling_mask[receptor_protected] = 1.0
            
            if not hasattr(self, '_protected_debug_printed'):
                protected_count = protected_mask.sum().item()
                print(f"[Creativity] protectedresidue: {protected_count}")
                print(f" - protectedresidue-receptor: unscaled")
                self._protected_debug_printed = True
        
        
        
        
        
        if bond_constraint_pairs is not None and bond_constraint_pairs.any():
            
            scaling_mask[bond_constraint_pairs] = 1.0  
            
            if not hasattr(self, '_bond_constraint_debug_printed'):
                bond_count = bond_constraint_pairs.sum().item()
                print(f"[Creativity] Bondvs: {bond_count}")
                print(f" - allbondvs(ligandandchainbondconstraint): unscaled")
                
                if ligand_mask is not None and ligand_mask.any():
                    ligand_ligand_pairs = ligand_mask[:, None] & ligand_mask[None, :]  # [L, L]
                    ligand_internal_bonds = bond_constraint_pairs & ligand_ligand_pairs
                    ligand_cross_bonds = bond_constraint_pairs & ~ligand_ligand_pairs
                    ligand_internal_count = ligand_internal_bonds.sum().item()
                    ligand_cross_count = ligand_cross_bonds.sum().item()
                    ligand_count = ligand_mask.sum().item()
                    print(f" - ligandresidue: {ligand_count}")
                    print(f" - Ligandbonds: {ligand_internal_count} (protectedligand)")
                    print(f" - chainbondconstraint: {ligand_cross_count} (protectedbondconstraint)")
                self._bond_constraint_debug_printed = True
        elif ligand_mask is not None and ligand_mask.any():
            
            ligand_any = ligand_mask[:, None] | ligand_mask[None, :]  # [L, L]
            scaling_mask[ligand_any] = 1.0  
            
            if not hasattr(self, '_ligand_debug_printed'):
                ligand_count = ligand_mask.sum().item()
                ligand_pairs_count = ligand_any.sum().item()
                print(f"[Creativity] ligandresidue: {ligand_count}")
                print(f" - allligandvs({ligand_pairs_count}): unscaled(mode)")
                self._ligand_debug_printed = True
        
        # ========================================
        
        # ========================================
        
        
        if scaling_residues is not None and len(scaling_residues) > 0:
            scaling_residue_mask = torch.zeros(L, dtype=torch.bool, device=device)
            for idx in scaling_residues:
                if 0 <= idx < L:
                    scaling_residue_mask[idx] = True
            
            if scaling_residue_mask.any():
                
                scaling_residue_2d = scaling_residue_mask[:, None] | scaling_residue_mask[None, :]  # [L, L]
                
                scaling_mask[scaling_residue_2d] = scale
                
                if not hasattr(self, '_scaling_residues_debug_printed'):
                    scaling_count = scaling_residue_mask.sum().item()
                    scaled_pairs_count = scaling_residue_2d.sum().item()
                    print(f"[Creativity] scalingresidue: {scaling_count}")
                    print(f" - scalingresidueandallresiduevs({scaled_pairs_count}): scaling(ligand)")
                    print(f" - scalingresiduelist(0-based): {scaling_residues}")
                    self._scaling_residues_debug_printed = True
        
        
        # scaling_mask: [L, L] -> [1, L, L, 1] for broadcasting
        scaling_mask = scaling_mask.view(1, L, L, 1)
        z_scaled = z * scaling_mask
        
        return z_scaled
    
    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        soft_sequence: Optional[torch.Tensor] = None,
        s_inputs: Optional[torch.Tensor] = None,  
        return_s_inputs: bool = False,  
        target_distogram: Optional[torch.Tensor] = None,  
        
        creativity: float = 0.0,  
        cdr_mask: Optional[torch.Tensor] = None,  
        receptor_mask: Optional[torch.Tensor] = None,  
        binder_mask: Optional[torch.Tensor] = None,  
        hotspot_indices: Optional[list] = None,  
        receptor_start: int = 0,  
        scaling_residues: Optional[List[int]] = None,  
        scaling_residues_only: bool = False,  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward."""
        
        
        
        if soft_sequence is not None:
            feats = feats.copy()  
            
            
            if soft_sequence.shape[-1] == 20:
                
                B, L, _ = soft_sequence.shape
                soft_seq_33 = torch.zeros(
                    B, L, 33,
                    dtype=soft_sequence.dtype,
                    device=soft_sequence.device
                )
                soft_seq_33[:, :, 2:22] = soft_sequence
                feats['res_type'] = soft_seq_33
            else:
                
                feats['res_type'] = soft_sequence
        
        # 1. Input embedding
        if s_inputs is None:
            s_inputs = self.input_embedder(feats)
        
        # =======================================================
        
        # =======================================================
        
        
        # 
        
        
        metadata = feats.get('metadata', {})
        
        
        if not hasattr(self, '_injection_debug_checked'):
            print(f"[PairformerWrapper] Injection condition check:")
            print(f"   feature_prep is not None: {self.feature_prep is not None}")
            print(f"   metadata.get('mode'): {metadata.get('mode')}")
            print(f"   metadata keys: {list(metadata.keys())}")
            self._injection_debug_checked = True
        
        if self.feature_prep is not None and metadata.get('mode') == 'complex':
            
            if not hasattr(self, '_injection_debug_printed'):
                print(f"[PairformerWrapper] Applying complex injection (mode={metadata.get('mode')})")
                print(f"[PairformerWrapper]   s_inputs shape before: {s_inputs.shape}")
                self._injection_debug_printed = True
            try:
                
                
                if hasattr(self.feature_prep, '_binder_soft_res_type'):
                    s_inputs = self.feature_prep.apply_injection(s_inputs, metadata, feats=feats)
                    
                    
                    if not hasattr(self, '_injection_debug_printed_after'):
                        print(f"[PairformerWrapper]   s_inputs shape after: {s_inputs.shape}")
                        print(f"[PairformerWrapper] Injection successful")
                        self._injection_debug_printed_after = True
                else:
                    
                    if not hasattr(self, '_injection_skip_init_printed'):
                        print(f"[PairformerWrapper] Skipping injection (initialization phase, no soft sequence yet)")
                        self._injection_skip_init_printed = True
            except Exception as e:
                print(f"[PairformerWrapper] Injection failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            if not hasattr(self, '_injection_skip_printed'):
                reason = []
                if self.feature_prep is None:
                    reason.append("feature_prep is None")
                if metadata.get('mode') != 'complex':
                    reason.append(f"mode='{metadata.get('mode')}' != 'complex'")
                print(f"[PairformerWrapper] Skipping injection: {', '.join(reason)}")
                self._injection_skip_printed = True
        # =======================================================
        
        
        s_init = self.s_init(s_inputs)
        
        
        z_init = (
            self.z_init_1(s_inputs)[:, :, None] +
            self.z_init_2(s_inputs)[:, None, :]
        )
        
        
        if not hasattr(self, '_z_init_debug_printed'):
            print(f"\n[DEBUG] ========== z_init Building Steps ==========")
            print(f"   z_init after broadcast: {z_init.shape}")
        
        
        relative_position_encoding = self.rel_pos(feats)
        if not hasattr(self, '_z_init_debug_printed'):
            print(f"   rel_pos output: {relative_position_encoding.shape}")
        z_init = z_init + relative_position_encoding
        if not hasattr(self, '_z_init_debug_printed'):
            print(f"   z_init after rel_pos: {z_init.shape}")
        
        
        token_bonds_out = self.token_bonds(feats["token_bonds"].float())
        if not hasattr(self, '_z_init_debug_printed'):
            print(f"   token_bonds output: {token_bonds_out.shape}")
        z_init = z_init + token_bonds_out
        if not hasattr(self, '_z_init_debug_printed'):
            print(f"   z_init after token_bonds: {z_init.shape}")
        
        
        
        if self.bond_type_feature and self.token_bonds_type is not None and "type_bonds" in feats:
            type_bonds_out = self.token_bonds_type(feats["type_bonds"].long())
            if not hasattr(self, '_z_init_debug_printed'):
                print(f"   type_bonds output: {type_bonds_out.shape}")
            z_init = z_init + type_bonds_out
            if not hasattr(self, '_z_init_debug_printed'):
                print(f"   z_init after type_bonds: {z_init.shape}")
        elif not hasattr(self, '_z_init_debug_printed'):
            if not self.bond_type_feature:
                print(f" bond_type_featurenot,skiptype_bonds embedding")
            elif self.token_bonds_type is None:
                print(f" token_bonds_typenotinit,skiptype_bonds embedding")
            elif "type_bonds" not in feats:
                print(f" featsintype_bonds,skiptype_bonds embedding")
        
        
        contact_cond = self.contact_conditioning(feats)
        if not hasattr(self, '_z_init_debug_printed'):
            print(f"   contact_conditioning output: {contact_cond.shape}")
        z_init = z_init + contact_cond
        if not hasattr(self, '_z_init_debug_printed'):
            print(f"   z_init final: {z_init.shape}")
            print(f"[DEBUG] ================================================\n")
            self._z_init_debug_printed = True
        
        
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        
        # Compute masks
        mask = feats["token_pad_mask"].float()
        
        
        if (self.use_interface_mask and 
            target_distogram is not None and 
            metadata and 
            metadata.get('mode') == 'complex'):
            
            receptor_slice = metadata.get('receptor_slice', slice(0, metadata['receptor_length']))
            binder_slice = metadata.get('binder_slice')
            
            if binder_slice is not None:
                
                interface_receptor_indices, interface_binder_indices = \
                    self.identify_interface_residues_from_distogram(
                        target_distogram, binder_slice, receptor_slice,
                        contact_cutoff=9.0, min_prob=0.1  
                    )
                
                
                if not hasattr(self, '_interface_mask_debug_printed'):
                    print(f"[InterfaceMask] interfacenotemaskalready")
                    print(f" - to {len(interface_receptor_indices)} interfaceresiduevs")
                    print(f" - interfacereceptorresidue: {len(torch.unique(interface_receptor_indices))}")
                    print(f" - interfacebinderresidue: {len(torch.unique(interface_binder_indices))}")
                    self._interface_mask_debug_printed = True
                
                
                interface_attention_mask = torch.zeros_like(mask)  # [B, L]
                
                
                interface_attention_mask[:, interface_receptor_indices] = 1.0
                
                
                interface_attention_mask[:, binder_slice] = 1.0
                
                
                interface_attention_mask[:, receptor_slice] = 1.0
                
                
                mask = mask * interface_attention_mask
        
        pair_mask = mask[:, :, None] * mask[:, None, :]
        
        
        if (self.use_interface_mask and 
            target_distogram is not None and 
            metadata and 
            metadata.get('mode') == 'complex' and
            interface_receptor_indices is not None and
            interface_binder_indices is not None):
            
            receptor_slice = metadata.get('receptor_slice', slice(0, metadata['receptor_length']))
            binder_slice = metadata.get('binder_slice')
            
            if binder_slice is not None:
                
                
                
                interface_pair_mask = torch.zeros_like(pair_mask)  # [B, L, L]
                
                
                for binder_idx, receptor_idx in zip(interface_binder_indices, interface_receptor_indices):
                    interface_pair_mask[:, binder_idx, receptor_idx] = 1.0
                    interface_pair_mask[:, receptor_idx, binder_idx] = 1.0
                
                
                interface_pair_mask[:, binder_slice, binder_slice] = 1.0
                
                
                interface_pair_mask[:, receptor_slice, receptor_slice] = 1.0
                
                
                pair_mask = pair_mask * interface_pair_mask
                
                
                if not hasattr(self, '_interface_pair_mask_debug_printed'):
                    mask_coverage = pair_mask.sum().item() / pair_mask.numel()
                    print(f"[InterfaceMask] interfacepairwise maskalready")
                    print(f" - Mask: {mask_coverage*100:.1f}%")
                    print(f" - residuevs: {pair_mask.sum().item():.0f} / {pair_mask.numel()}")
                    self._interface_pair_mask_debug_printed = True
        
        for i in range(self.num_recycles + 1):
            
            is_last_recycle = (i == self.num_recycles)
            
            with torch.set_grad_enabled(is_last_recycle):
                
                if i == 0 and not hasattr(self, '_recycle_debug_printed'):
                    print(f"[DEBUG] Recycling step {i}:")
                    print(f"   z before recycle: {z.shape}")
                    z_normed = self.z_norm(z.detach() if not is_last_recycle else z)
                    print(f"   z_norm output: {z_normed.shape}")
                    z_recycled = self.z_recycle(z_normed)
                    print(f"   z_recycle output: {z_recycled.shape}")
                    print(f"   z_init: {z_init.shape}")
                    self._recycle_debug_printed = True
                
                
                if z.dim() == 5 and z.shape[1] == 1:
                    z = z.squeeze(1)
                
                # Apply recycling
                s = s_init + self.s_recycle(self.s_norm(s.detach() if not is_last_recycle else s))
                z_recycled_output = self.z_recycle(self.z_norm(z.detach() if not is_last_recycle else z))
                
                
                if z_recycled_output.dim() == 5 and z_recycled_output.shape[1] == 1:
                    z_recycled_output = z_recycled_output.squeeze(1)
                
                z = z_init + z_recycled_output
                
                
                if i == 0 and not hasattr(self, '_z_after_recycle_printed'):
                    print(f"[DEBUG] z IMMEDIATELY after recycling update: {z.shape}")
                    if z.dim() == 5:
                        print(f" WARNING: z became 5-dimensional!")
                        print(f"   z_init.shape = {z_init.shape}")
                        z_recycle_out = self.z_recycle(self.z_norm(torch.zeros_like(z_init)))
                        print(f"   z_recycle(z_norm(zeros)).shape = {z_recycle_out.shape}")
                        print(f"   Attempting to squeeze extra dimension...")
                    self._z_after_recycle_printed = True
                
                
                if i == 0 and not hasattr(self, '_z_before_squeeze_printed'):
                    print(f"[DEBUG] z before squeeze check: {z.shape}, z.dim()={z.dim()}")
                    self._z_before_squeeze_printed = True
                
                if z.dim() == 5 and z.shape[1] == 1:
                    z = z.squeeze(1)
                    if not hasattr(self, '_z_squeeze_warned'):
                        print(f"[FIX] Squeezed z from 5D to 4D: {z.shape}")
                        self._z_squeeze_warned = True
                
                if i == 0 and not hasattr(self, '_z_after_squeeze_printed'):
                    print(f"[DEBUG] z after squeeze check: {z.shape}")
                    self._z_after_squeeze_printed = True
                
                
                if self.use_templates and self.template_module is not None:
                    z = z + self.template_module(
                        z, feats, pair_mask, use_kernels=False
                    )
                
                if i == 0 and not hasattr(self, '_z_after_template_printed'):
                    print(f"[DEBUG] z after template check: {z.shape}")
                    self._z_after_template_printed = True
                
                
                if not hasattr(self, '_z_dim_debug_printed'):
                    print(f"\n[DEBUG] ========== Tensor Dimensions Check ==========")
                    print(f"   z shape: {z.shape} (expected: [B, L, L, D_z])")
                    print(f"   z_init shape: {z_init.shape}")
                    print(f"   s shape: {s.shape}")
                    print(f"   s_init shape: {s_init.shape}")
                    print(f"   s_inputs shape: {s_inputs.shape}")
                    print(f"   mask shape: {mask.shape}")
                    print(f"   pair_mask shape: {pair_mask.shape}")
                    print(f"\n   MSA-related feats:")
                    for key in ['msa', 'msa_mask', 'msa_paired', 'has_deletion', 'deletion_value']:
                        if key in feats:
                            print(f"   feats['{key}'] shape: {feats[key].shape}")
                        else:
                            print(f"   feats['{key}']: NOT FOUND")
                    print(f"\n   Other feats:")
                    print(f"   feats['token_pad_mask'] shape: {feats['token_pad_mask'].shape}")
                    if 'token_bonds' in feats:
                        print(f"   feats['token_bonds'] shape: {feats['token_bonds'].shape}")
                    print(f"[DEBUG] ================================================\n")
                    self._z_dim_debug_printed = True
                
                
                if z.dim() == 5:
                    if z.shape[1] == 1:
                        z = z.squeeze(1)
                        if not hasattr(self, '_z_final_squeeze_warned'):
                            print(f"[FIX] Final squeeze before MSA: z.shape = {z.shape}")
                            self._z_final_squeeze_warned = True
                    else:
                        raise RuntimeError(f"z has unexpected 5D shape: {z.shape}. Cannot fix automatically.")
                
                # MSA module
                z = z + self.msa_module(
                    z, s_inputs, feats, use_kernels=False
                )
                
                
                if self.use_checkpointing and is_last_recycle:
                    
                    import torch.utils.checkpoint as checkpoint_utils
                    
                    for layer in self.pairformer_module.layers:
                        def run_layer(s_in, z_in, m_in, pm_in):
                            return layer(s_in, z_in, m_in, pm_in, None, False)  # use_kernels=False
                        
                        s, z = checkpoint_utils.checkpoint(
                            run_layer,
                            s, z, mask, pair_mask,
                            use_reentrant=False
                        )
                else:
                    
                    s, z = self.pairformer_module(
                        s, z,
                        mask=mask,
                        pair_mask=pair_mask,
                        use_kernels=False
                    )
                
                
                
                if creativity > 0.0:
                    
                    bond_constraint_pairs = None
                    if 'token_bonds' in feats and feats['token_bonds'] is not None:
                        token_bonds_feat = feats['token_bonds']  
                        if token_bonds_feat.dim() == 4:
                            token_bonds_feat = token_bonds_feat.squeeze(0).squeeze(-1)  # [L, L]
                        elif token_bonds_feat.dim() == 3:
                            token_bonds_feat = token_bonds_feat.squeeze(-1)  # [L, L]
                        
                        bond_constraint_pairs = (token_bonds_feat > 0)  # [L, L] bool
                    
                    z = self.apply_creativity_scaling(
                        z=z,
                        creativity=creativity,
                        cdr_mask=cdr_mask,
                        receptor_mask=receptor_mask,
                        binder_mask=binder_mask,
                        hotspot_indices=hotspot_indices,
                        receptor_start=receptor_start,
                        bond_constraint_pairs=bond_constraint_pairs,  
                        scaling_residues=scaling_residues,  
                        scaling_residues_only=scaling_residues_only,  
                    )
            
                
                
                
            
            
            
            if self.monitor_z and is_last_recycle:
                z_stats = self.get_z_stats(z)
                self.z_stats_history.append(z_stats)
                
                
                
                # if z_stats['z_max'] > 50 or z_stats['z_min'] < -50:
                
        
        
        if return_s_inputs:
            return s, z, s_inputs
        else:
            return s, z
    
    def get_z_stats(self, z: torch.Tensor) -> Dict[str, float]:
        """get z stats."""
        with torch.no_grad():
            return {
                'z_min': z.min().item(),
                'z_max': z.max().item(),
                'z_mean': z.mean().item(),
                'z_std': z.std().item(),
                'z_abs_max': z.abs().max().item()
            }
    
    def get_frozen_params_count(self) -> Dict[str, int]:
        """get frozen params count."""
        s_recycle_params = sum(p.numel() for p in self.s_recycle.parameters())
        z_recycle_params = sum(p.numel() for p in self.z_recycle.parameters())
        
        return {
            's_recycle_params': s_recycle_params,
            'z_recycle_params': z_recycle_params,
            'total_frozen': s_recycle_params + z_recycle_params
        }
    
    def reset_z_stats_history(self):
        """reset z stats history."""
        self.z_stats_history = []
    
    def __repr__(self):
        """  repr  ."""
        frozen_status = 'frozen' if self.freeze_recycle_layers else 'not frozen'
        frozen_params = self.get_frozen_params_count()
        
        return (f"PairformerWrapper(\n"
                f"  num_recycles={self.num_recycles},\n"
                f"  recycling_layers={frozen_status},\n"
                f"  frozen_params={frozen_params['total_frozen']:,},\n"
                f"  monitor_z={self.monitor_z}\n"
                f")")


# ============================================================================

# ============================================================================

def load_pairformer_wrapper(
    boltz_model_path: str,
    num_recycles: int = 3,
    freeze_recycle: bool = True,
    device: str = 'cuda'
) -> PairformerWrapper:
    """load pairformer wrapper."""
    
    
    raise NotImplementedError('load')

