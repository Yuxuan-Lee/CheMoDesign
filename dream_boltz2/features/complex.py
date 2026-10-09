"""Build Boltz-2 features for a receptor-binder complex, including ligands and covalent bonds."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List  
import warnings
import numpy as np  


# ============================================================================

# ============================================================================

STANDARD_AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'  
SUPER_RESIDUE = 'X'  


BOLTZ_FEATURIZER_DEFAULT_MAX_SEQS = 16384


MODIFIED_RESIDUE_MAP = {
    
    'SEP': 'S',  
    'TPO': 'T',  
    'PTR': 'Y',  
    
    
    'DAL': 'A',  # D-Ala
    'DAR': 'R',  # D-Arg
    'DSN': 'N',  # D-Asn
    'DSP': 'D',  # D-Asp
    'DCY': 'C',  # D-Cys
    'DGL': 'E',  # D-Glu
    'DGN': 'Q',  # D-Gln
    'DHI': 'H',  # D-His
    'DIL': 'I',  # D-Ile
    'DLE': 'L',  # D-Leu
    'DLY': 'K',  # D-Lys
    'MED': 'M',  # D-Met
    'DPN': 'F',  # D-Phe
    'DPR': 'P',  # D-Pro
    'DSE': 'S',  # D-Ser
    'DTH': 'T',  # D-Thr
    'DTR': 'W',  # D-Trp
    'DTY': 'Y',  # D-Tyr
    'DVA': 'V',  # D-Val
    
    
    'MSE': 'M',  # Selenomethionine
    'HYP': 'P',  # Hydroxyproline
    'MLY': 'K',  # N-methyl-lysine
    'M3L': 'K',  # N-trimethyl-lysine
    
    
    
}


class FeaturePreparatorComplex:
    """FeaturePreparatorComplex."""
    
    def __init__(
        self,
        boltz_model,
        device: str = 'cuda',
        mode: str = 'monomer',
        enable_soft_atoms: bool = False,
        soft_basis_mode: str = "absolute",
        isolated_frame: str = "n_ca_c",
    ):
        """  init  ."""
        self.model = boltz_model
        self.device = device
        self.mode = mode
        self.enable_soft_atoms = enable_soft_atoms
        if soft_basis_mode not in ("delta", "absolute"):
            raise ValueError(f"unknown soft_basis_mode={soft_basis_mode!r}")
        self.soft_basis_mode = soft_basis_mode
        self.isolated_frame = isolated_frame
        self._candidate_basis = None  # CandidateTokenBasis
        self._last_feats = None
        
        
        self.receptor_length = None
        self.binder_length = None
        self.binder_slice = None  # slice(start, end)
        
        
        self._hard_feats = None  
        self._hard_s_inputs = None  
        
        print(f"[FeaturePreparator] init")
        print(f" mode: {mode}")
        print(f": {device}")
        print(f"  soft_atoms(token chemistry basis): {enable_soft_atoms}")
        if enable_soft_atoms:
            from dream_boltz2.model.soft_chemistry import inspect_embedder_flags, install_embedder_capture
            install_embedder_capture(boltz_model.input_embedder)
            flags = inspect_embedder_flags(boltz_model)
            print(f"  embedder flags: add_modified={flags.get('add_modified_flag')} "
                  f"use_residue_feats_atoms={flags.get('use_residue_feats_atoms')}")
            print(f"  soft_basis_mode={soft_basis_mode}  isolated_frame={isolated_frame}")
        self._hard_validation_mode = False

    def begin_hard_validation(self) -> None:
        """begin hard validation."""
        self._hard_validation_mode = True
        for attr in ("_binder_soft_res_type", "_binder_soft_sequence_56", "_binder_hard_idx"):
            if hasattr(self, attr):
                delattr(self, attr)
        print("[HardValidation] soft chemistry injection disabled; using argmax CCD only")

    def end_hard_validation(self) -> None:
        """end hard validation."""
        self._hard_validation_mode = False

    def _print_binder_msa_once(self, feats: Dict, binder_slice: slice) -> None:
        if getattr(self, "_printed_binder_msa", False):
            return
        self._printed_binder_msa = True
        try:
            sl = binder_slice
            msa = feats["msa"]
            msa_mask = feats["msa_mask"]
            profile = feats["profile"]
            deletion_mean = feats["deletion_mean"]
            print("[SoftChemistry] binder empty-MSA dump (featurizer default, not overwritten by p):")
            print(f"  msa.shape={tuple(msa.shape)}  msa_mask.shape={tuple(msa_mask.shape)}")
            print(f"  profile.shape={tuple(profile.shape)}  deletion_mean.shape={tuple(deletion_mean.shape)}")
            if msa.dim() == 3:
                print(f"  binder msa (query row): {msa[0, 0, sl].detach().cpu().tolist()}")
            if msa_mask.dim() == 3:
                print(f"  binder msa_mask (query row): {msa_mask[0, 0, sl].detach().cpu().tolist()}")
            elif msa_mask.dim() == 2:
                print(f"  binder msa_mask: {msa_mask[0, sl].detach().cpu().tolist()}")
            prof = profile[0, sl]
            print(f"  binder profile argmax: {prof.argmax(dim=-1).detach().cpu().tolist()}")
            print(f"  binder profile row-sum: {prof.sum(dim=-1).detach().cpu().tolist()}")
            dm = deletion_mean[0, sl] if deletion_mean.dim() >= 2 else deletion_mean[sl]
            print(f"  binder deletion_mean: {dm.detach().cpu().tolist()}")
            res = feats["res_type"][0, sl].float()
            n = min(prof.shape[-1], res.shape[-1])
            same = torch.allclose(prof[..., :n], res[..., :n], atol=1e-5)
            print(f"  profile matches hard res_type one-hot (empty-MSA query row): {bool(same)}")
        except Exception as e:
            print(f"[SoftChemistry] binder MSA dump failed: {e}")
    
    # ========================================================================
    
    # ========================================================================
    
    def prepare_monomer_features(
        self,
        length: int,
        is_cyclic: bool = False,
        constraints: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """prepare monomer features."""
        import tempfile
        import pickle
        import numpy as np
        from pathlib import Path
        
        
        template_seq = SUPER_RESIDUE * length
        
        print(f"[FeaturePrep] Generating features for sequence (length={length})...")
        
        
        cyclic_line = "\n    cyclic: true" if is_cyclic else ""
        yaml_content = f"""version: 1
sequences:
- protein:
    id: A
    sequence: {template_seq}
    msa: empty{cyclic_line}
"""
        temp_yaml = Path(tempfile.gettempdir()) / f"temp_monomer_{id(template_seq)}.yaml"
        temp_yaml.write_text(yaml_content)
        
        try:
            
            if not hasattr(self, 'tokenizer'):
                self._init_boltz_components()
            
            
            from boltz.data.parse.yaml import parse_yaml
            
            from dream_boltz2.io.quiet import suppress_all_boltz2_output
            
            with suppress_all_boltz2_output():
                target = parse_yaml(temp_yaml, self.ccd, self.mol_dir, boltz2=True)
            
            
            from boltz.data.types import Input
            input_data = Input(
                structure=target.structure,
                msa={},  
                record=target.record,
                residue_constraints=target.residue_constraints,
                templates=target.templates,
                extra_mols=target.extra_mols or {},
            )
            
            # 4. Tokenize
            tokenized = self.tokenizer.tokenize(input_data)
            
            # 5. Featurize
            molecules = {}
            molecules.update(self.canonicals)
            if input_data.extra_mols:
                molecules.update(input_data.extra_mols)
            if getattr(self, "_bundled_mols", None):
                molecules.update(self._bundled_mols)
            
            random = np.random.default_rng(42)
            
            feats = self.featurizer.process(
                tokenized,
                molecules=molecules,
                random=random,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=1,  
                pad_to_max_seqs=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=None,
                inference_contact_constraints=None,
                compute_constraint_features=True,
                override_method=None,
                compute_affinity=False,
            )
            
            print(f"   [OK] Features generated using Boltz2 official pipeline")
            
            
            feats_batched = {}
            for key, value in feats.items():
                if isinstance(value, torch.Tensor):
                    
                    value_batched = value.unsqueeze(0).to(self.device)
                    
                    
                    if key == 'template_mask' and value_batched.dim() == 3:
                        value_batched = value_batched.unsqueeze(-1)  # [B, T, L] -> [B, T, L, 1]
                    
                    feats_batched[key] = value_batched
                else:
                    feats_batched[key] = value
            
            
            feats_batched = self._detach_non_sequence_features(feats_batched)
            
            
            self._hard_feats = feats_batched
            
            metadata = {
                'length': length,
                'mode': 'monomer',
                'template': template_seq,
                'is_cyclic': is_cyclic,
                
                'binder_slice': slice(0, length),
                'binder_length': length
            }
            
            return feats_batched, metadata
            
        finally:
            
            if temp_yaml.exists():
                temp_yaml.unlink()
    
    def _init_boltz_components(self):
        """ init boltz components."""
        import pickle
        from pathlib import Path
        
        print("[FeaturePrep] Initializing Boltz2 components (first time)...")

        from dream_boltz2.assets import resolve_boltz_cache, resolve_ccd_pkl, resolve_mol_dir

        cache_dir = resolve_boltz_cache()
        self.mol_dir = resolve_mol_dir(cache_dir)
        if not (self.mol_dir / "ALA.pkl").is_file():
            raise FileNotFoundError(
                "Boltz-2 molecule library not found "
                f"(looked for {self.mol_dir / 'ALA.pkl'}). "
                "Run: python -m dream_boltz2.cli.setup_data"
            )

        ccd_path = resolve_ccd_pkl(cache_dir)
        if ccd_path is not None:
            with open(ccd_path, "rb") as f:
                self.ccd = pickle.load(f)
            print(f"   [OK] CCD loaded ({len(self.ccd)} entries) from {ccd_path}")
        else:
            self.ccd = {}
            print("   [OK] No ccd.pkl; components load from the molecule library")

        self._bundled_mols = {}
        ligand_dir = Path(__file__).resolve().parents[2] / "examples" / "ligands"
        if ligand_dir.is_dir():
            for pkl_path in sorted(ligand_dir.glob("*.pkl")):
                with open(pkl_path, "rb") as handle:
                    mol = pickle.load(handle)
                self._bundled_mols[pkl_path.stem] = mol
                self.ccd[pkl_path.stem] = mol
            if self._bundled_mols:
                print(f"   [OK] Bundled ligands: {', '.join(sorted(self._bundled_mols))}")

        from boltz.data.mol import load_canonicals, load_molecules
        self.canonicals = load_canonicals(self.mol_dir)
        
        
        
        
        modified_residues = [
            
            'DAL', 'DAR', 'DSN', 'DSP', 'DCY', 'DGL', 'DGN', 'DHI', 'DIL', 'DLE', 
            'DLY', 'MED', 'DPN', 'DPR', 'DSE', 'DTH', 'DTR', 'DTY', 'DVA',
            
            'SEP', 'TPO', 'PTR',
            
            'MLY', 'MLZ', 'M3L', 'ALY', '2MR', 'MME', 'MHS',
            
            'HYP', 'AIB', 'NLE', 'NVA', 'ORN', 'CIR', 'SAR', 'MSE',
            
            'SEC',
        ]
        
        
        import pickle
        loaded_modified = []
        missing_modified = []
        
        for res in modified_residues:
            mol_path = self.mol_dir / f"{res}.pkl"
            if mol_path.exists():
                try:
                    with open(mol_path, 'rb') as f:
                        self.canonicals[res] = pickle.load(f)
                    loaded_modified.append(res)
                except Exception as e:
                    print(f"   [ERROR] Failed to load {res}: {e}")
                    missing_modified.append(res)
            else:
                
                missing_modified.append(res)
        
        print(f"   [OK] Canonicals loaded: 20 standard + {len(loaded_modified)}/{len(modified_residues)} modified")
        
        if len(missing_modified) > 0:
            print(f"   [WARNING] Missing molecule files ({len(missing_modified)}):")
            print(f"             {', '.join(missing_modified)}")
            print(f"   [INFO] These residues will NOT be available for optimization")
            print(f"   [INFO] To use them, ensure .pkl files exist in: {self.mol_dir}")
        
        if len(loaded_modified) > 0:
            print(f"   [OK] Available modified residues:")
            
            d_amino = [r for r in loaded_modified if r in ['DAL', 'DAR', 'DSN', 'DSP', 'DCY', 'DGL', 'DGN', 'DHI', 'DIL', 'DLE', 'DLY', 'MED', 'DPN', 'DPR', 'DSE', 'DTH', 'DTR', 'DTY', 'DVA']]
            phospho = [r for r in loaded_modified if r in ['SEP', 'TPO', 'PTR']]
            methyl = [r for r in loaded_modified if r in ['MLY', 'MLZ', 'M3L', 'ALY', '2MR', 'MME', 'MHS']]
            special = [r for r in loaded_modified if r in ['HYP', 'AIB', 'NLE', 'NVA', 'ORN', 'CIR', 'SAR', 'MSE', 'SEC']]
            
            if d_amino:
                print(f"        D-amino acids ({len(d_amino)}): {', '.join(d_amino)}")
            if phospho:
                print(f"        Phosphorylated ({len(phospho)}): {', '.join(phospho)}")
            if methyl:
                print(f"        Methylated ({len(methyl)}): {', '.join(methyl)}")
            if special:
                print(f"        Special ({len(special)}): {', '.join(special)}")
        
        
        from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
        from boltz.data.feature.featurizerv2 import Boltz2Featurizer
        
        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        print(f"   [OK] Tokenizer and Featurizer created")
    
    def _find_cache_dir(self):
        """Return the Boltz cache that contains the molecule library."""
        from dream_boltz2.assets import resolve_boltz_cache, resolve_mol_dir

        cache = resolve_boltz_cache()
        mols_dir = resolve_mol_dir(cache)
        print(f"   [OK] Boltz cache: {cache}")
        if mols_dir.is_dir():
            print(f"   [OK] Molecule library: {mols_dir}")
        else:
            print(f"   [WARNING] Molecule library not found: {mols_dir}")
        return cache
    
    def _generate_complex_features_with_msa(
        self,
        receptor_sequence: str,
        binder_sequence: str = None,  
        binder_chains: Optional[List[Tuple[str, str]]] = None,  
        receptor_msa_file: Optional[str] = None,  
        binder_msa_file: Optional[str] = None,  
        max_msa_seqs: Optional[int] = None,
        binder_cyclic: bool = False,
        binder_modifications: List[Dict] = None,  
        contact_constraints: Optional[List[Tuple[int, int, float]]] = None,  
        ligands: Optional[List[Dict]] = None,  
        template_file: Optional[str] = None,  
        template_chain_id: Optional[str] = None,  
        template_id: Optional[str] = None,  
        template_force: bool = False,  
        template_threshold: float = 3.0,  
        yaml_constraints: Optional[List] = None,  
        receptor_chain_id: Optional[str] = None,  
        binder_chain_id: Optional[str] = None,  
    ) -> Dict:
        """ generate complex features with msa."""
        import tempfile
        import numpy as np
        from pathlib import Path
        
        
        if not hasattr(self, '_complex_feature_first_print'):
            print(f"[FeaturePrep] Generating complex features with MSA...")
            print(f"  Receptor: {len(receptor_sequence)} residues")
            if binder_chains is not None and len(binder_chains) > 0:
                total_binder = sum(len(seq) for _, seq in binder_chains)
                print(f"  Binder: {len(binder_chains)} chains, {total_binder} total residues")
            elif binder_sequence:
                print(f"  Binder: {len(binder_sequence)} residues")
            self._complex_feature_first_print = True
        
        
        
        if receptor_msa_file:
            receptor_msa_line = f"    msa: {receptor_msa_file}"
        else:
            receptor_msa_line = "    msa: empty"  
        
        
        if binder_msa_file:
            binder_msa_line = f"    msa: {binder_msa_file}"
        else:
            binder_msa_line = "    msa: empty"  
        
        
        if binder_chains is not None and len(binder_chains) > 0:
            
            binder_sequences_yaml = ""
            for chain_id, chain_seq in binder_chains:
                
                binder_cyclic_line = "\n    cyclic: true" if binder_cyclic else ""  
                binder_sequences_yaml += f"""
- protein:
    id: {chain_id}
    sequence: {chain_seq}
{binder_msa_line}{binder_cyclic_line}"""
            
        else:
            
            if binder_sequence is None:
                raise ValueError("Either binder_sequence or binder_chains must be provided")
            
            binder_cyclic_line = "\n    cyclic: true" if binder_cyclic else ""  
            
            
            binder_mods_lines = ""
            if binder_modifications and len(binder_modifications) > 0:
                binder_mods_lines = "\n    modifications:"
                for mod in binder_modifications:
                    binder_mods_lines += f"\n      - position: {mod['position']}"
                    binder_mods_lines += f"\n        ccd: \"{mod['ccd']}\""
            
            
            
            
            binder_chain = binder_chain_id if binder_chain_id is not None else 'A'
            binder_sequences_yaml = f"""
- protein:
    id: {binder_chain}
    sequence: {binder_sequence}
{binder_msa_line}{binder_cyclic_line}{binder_mods_lines}"""
        
        
        constraints_lines = ""
        has_constraints = False
        
        
        if yaml_constraints and len(yaml_constraints) > 0:
            if not has_constraints:
                constraints_lines = "\nconstraints:"
                has_constraints = True
            
            for constraint in yaml_constraints:
                constraint_type = constraint.constraint_type.lower()
                constraint_data = constraint.data
                
                if constraint_type == "bond":
                    
                    
                    if isinstance(constraint_data, dict):
                        atom1 = constraint_data.get("atom1", [])
                        atom2 = constraint_data.get("atom2", [])
                    elif isinstance(constraint_data, list) and len(constraint_data) >= 2:
                        
                        atom1 = constraint_data[0] if len(constraint_data) > 0 else []
                        atom2 = constraint_data[1] if len(constraint_data) > 1 else []
                    else:
                        atom1 = []
                        atom2 = []
                    
                    if len(atom1) == 3 and len(atom2) == 3:
                        constraints_lines += f"""
- bond:
    atom1: [{atom1[0]}, {atom1[1]}, {atom1[2]}]
    atom2: [{atom2[0]}, {atom2[1]}, {atom2[2]}]"""
                        if not hasattr(self, '_yaml_bond_constraint_printed'):
                            print(f" [CONSTRAINTS] addbondconstraint: {atom1} <-> {atom2}")
                            self._yaml_bond_constraint_printed = True
                
                elif constraint_type == "contact":
                    
                    token1 = constraint_data.get("token1", [])
                    token2 = constraint_data.get("token2", [])
                    max_distance = constraint_data.get("max_distance", 6.0)
                    force = constraint_data.get("force", False)
                    if len(token1) == 2 and len(token2) == 2:
                        constraints_lines += f"""
- contact:
    token1: [{token1[0]}, {token1[1]}]
    token2: [{token2[0]}, {token2[1]}]
    max_distance: {max_distance}
    force: {str(force).lower()}"""
                        if not hasattr(self, '_yaml_contact_constraint_printed'):
                            print(f" [CONSTRAINTS] addcontactconstraint: {token1} <-> {token2}, max_distance={max_distance}Å")
                            self._yaml_contact_constraint_printed = True
                
                elif constraint_type == "pocket":
                    
                    binder = constraint_data.get("binder", "")
                    contacts = constraint_data.get("contacts", [])
                    max_distance = constraint_data.get("max_distance", 6.0)
                    force = constraint_data.get("force", False)
                    if binder and contacts:
                        contacts_str = ", ".join([f"[{c[0]}, {c[1]}]" for c in contacts])
                        constraints_lines += f"""
- pocket:
    binder: {binder}
    contacts: [{contacts_str}]
    max_distance: {max_distance}
    force: {str(force).lower()}"""
                        if not hasattr(self, '_yaml_pocket_constraint_printed'):
                            print(f" [CONSTRAINTS] addpocketconstraint: binder={binder}, contacts={len(contacts)}")
                            self._yaml_pocket_constraint_printed = True
        
        
        if contact_constraints and len(contact_constraints) > 0:
            if not has_constraints:
                constraints_lines = "\nconstraints:"
                has_constraints = True
            for binder_res_idx, receptor_res_idx, max_distance in contact_constraints:
                
                
                if binder_chains is not None and len(binder_chains) > 0:
                    
                    current_pos = 0
                    binder_chain_id = None
                    binder_chain_res_idx = None
                    for chain_id, chain_seq in binder_chains:
                        chain_len = len(chain_seq)
                        if current_pos <= binder_res_idx < current_pos + chain_len:
                            binder_chain_id = chain_id
                            binder_chain_res_idx = binder_res_idx - current_pos
                            break
                        current_pos += chain_len
                    if binder_chain_id is None:
                        
                        binder_chain_id = binder_chains[0][0]
                        binder_chain_res_idx = binder_res_idx
                else:
                    
                    binder_chain_id = 'A'
                    binder_chain_res_idx = binder_res_idx
                
                constraints_lines += f"""
- contact:
    token1: [{binder_chain_id}, {binder_chain_res_idx + 1}]
    token2: [B, {receptor_res_idx + 1}]
    max_distance: {max_distance}
    force: true"""
        
        
        ligand_lines = ""
        if ligands and len(ligands) > 0:
            for ligand in ligands:
                ligand_id = ligand.get('id', 'C')  
                if 'ccd' in ligand:
                    
                    ligand_lines += f"""
- ligand:
    id: {ligand_id}
    ccd: {ligand['ccd']}"""
                elif 'smiles' in ligand:
                    
                    ligand_lines += f"""
- ligand:
    id: {ligand_id}
    smiles: '{ligand['smiles']}'"""
        
        
        template_lines = ""
        if template_file and template_chain_id:
            template_path = Path(template_file)
            if not template_path.exists():
                print(f"[WARNING] Templatefilenotexists: {template_file},skiptemplateconstraint")
            else:
                
                if template_file.endswith('.cif') or template_file.endswith('.mmcif'):
                    template_key = "cif"
                elif template_file.endswith('.pdb'):
                    template_key = "pdb"
                else:
                    
                    template_key = "pdb"
                    print(f"[WARNING] Templatefilenot: {template_file},asPDB")
                
                
                if template_id is None:
                    
                    if template_key == "pdb":
                        template_id_val = f"{template_chain_id}1"
                    else:
                        
                        template_id_val = template_chain_id
                else:
                    template_id_val = template_id
                
                
                template_lines = f"""
templates:
  - {template_key}: {template_file}
    chain_id: [{template_chain_id}]
    template_id: [{template_id_val}]"""
                
                
                if template_force:
                    template_lines += f"""
    force: true
    threshold: {template_threshold}"""
        
        
        
        receptor_chain = receptor_chain_id if receptor_chain_id is not None else 'B'
        
        
        
        yaml_content = f"""version: 1
sequences:
- protein:
    id: {receptor_chain}
    sequence: {receptor_sequence}
{receptor_msa_line}{binder_sequences_yaml}{ligand_lines}{constraints_lines}{template_lines}
"""
        
        
        if binder_chains is not None and len(binder_chains) > 0:
            chain_ids_str = '_'.join(chain_id for chain_id, _ in binder_chains)
            temp_yaml = Path(tempfile.gettempdir()) / f"complex_{id(receptor_sequence)}_{chain_ids_str}.yaml"
        else:
            temp_yaml = Path(tempfile.gettempdir()) / f"complex_{id(receptor_sequence)}_{id(binder_sequence)}.yaml"
        temp_yaml.write_text(yaml_content)
        
        try:
            
            if not hasattr(self, 'ccd') or not hasattr(self, 'mol_dir'):
                self._init_boltz_components()
            
            
            from boltz.data.parse.yaml import parse_yaml
            
            if not hasattr(self, '_yaml_debug_printed'):
                print(f"   [DEBUG] YAML content (first 500 chars):\n{str(yaml_content)[:500]}...")
                self._yaml_debug_printed = True
            
            
            
            from dream_boltz2.io.quiet import suppress_all_boltz2_output
            
            with suppress_all_boltz2_output():
                try:
                    target = parse_yaml(temp_yaml, self.ccd, self.mol_dir, boltz2=True)
                except KeyError as e:
                    
                    if 'atom' in str(e).lower() or 'idx' in str(e).lower():
                        print(f"\n [ERROR] Bondconstraintparsefailed: {e}")
                        print(f" [ERROR]:")
                        print(f" 1. ligandatomnot(checkYAMLinatom2atom,'C5')")
                        print(f" 2. ligandresidueindexnot(YAMLinuse1-based,[C, 1, C5])")
                        print(f" 3. ligandchainIDnot(checkYAMLinligand id,'C')")
                        print(f" [ERROR] checkYAMLconfiginbondconstraint")
                    raise
            
            
            if not hasattr(self, '_ligand_bond_debug_printed'):
                if hasattr(target, 'structure') and hasattr(target.structure, 'bonds'):
                    bonds = target.structure.bonds
                    if bonds is not None and len(bonds) > 0:
                        
                        ligand_bonds = []
                        for bond in bonds:
                            
                            
                            ligand_bonds.append(bond)
                        
                        
                        from collections import Counter
                        bond_types = Counter()
                        for bond in bonds:
                            
                            try:
                                if isinstance(bond, np.ndarray) and bond.dtype.names:
                                    # numpy structured array
                                    bond_type = bond["type"]
                                elif hasattr(bond, '__getitem__'):
                                    
                                    bond_type = bond["type"]
                                else:
                                    
                                    bond_type = getattr(bond, 'type', -1)
                            except (KeyError, IndexError, TypeError):
                                bond_type = -1
                            
                            
                            bond_type_names = {0: 'OTHER', 1: 'SINGLE', 2: 'DOUBLE', 3: 'TRIPLE', 4: 'AROMATIC', 5: 'COVALENT'}
                            bond_type_name = bond_type_names.get(int(bond_type), f'UNKNOWN({bond_type})')
                            bond_types[bond_type_name] += 1
                        
                        if bond_types:
                            print(f" [LIGAND BONDS DEBUG] to {len(bonds)}:")
                            for bond_type_name, count in bond_types.most_common():
                                print(f"      {bond_type_name}: {count} ")
                            if bond_types.get('AROMATIC', 0) == 0:
                                print(f" [LIGAND BONDS DEBUG] warning:nottoAROMATIC！")
                                print(f" thisligandnot()")
                                print(f":checkCCDfilewhethersaveinfo")
                                
                                try:
                                    from boltz.data.mol import load_molecules
                                    mols_dict = {}
                                    
                                    mol_dir = str(self.mol_dir) if hasattr(self, 'mol_dir') else None
                                    if mol_dir:
                                        for lig_info in ligands:
                                            ccd = lig_info.get('ccd', '')
                                            if ccd:
                                                bundled = getattr(self, "_bundled_mols", {})
                                                if ccd in bundled:
                                                    mols_dict[ccd] = bundled[ccd]
                                                else:
                                                    mols_dict[ccd] = load_molecules(mol_dir, [ccd])[ccd]
                                                print(f" [LIGAND BONDS DEBUG] checkCCDfile {ccd}:")
                                                for bond in mols_dict[ccd].GetBonds():
                                                    bond_type = bond.GetBondType()
                                                    bond_type_name = bond_type.name
                                                    is_aromatic = bond.GetIsAromatic()
                                                    print(f" {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}: type={bond_type}, name='{bond_type_name}', is_aromatic={is_aromatic}")
                                                    
                                                    from boltz.data import const
                                                    if bond_type_name in const.bond_type_ids:
                                                        print(f" inBoltz2in: {const.bond_type_ids[bond_type_name]}")
                                                    else:
                                                        print(f" notinBoltz2in！: {list(const.bond_type_ids.keys())}")
                                    else:
                                        print(f" [LIGAND BONDS DEBUG] nogetmol_dir")
                                except Exception as e:
                                    import traceback
                                    print(f" [LIGAND BONDS DEBUG] failed: {e}")
                                    print(f" [LIGAND BONDS DEBUG] error: {traceback.format_exc()}")
                            else:
                                print(f" [LIGAND BONDS DEBUG] to {bond_types['AROMATIC']} AROMATIC")
                self._ligand_bond_debug_printed = True
            
            
            msa_dict = {}
            if hasattr(target, 'record') and target.record and hasattr(target.record, 'chains'):
                from boltz.data.parse.a3m import parse_a3m
                
                for chain_info in target.record.chains:
                    chain_id = chain_info.chain_id  
                    msa_path = chain_info.msa_id
                    
                    if isinstance(msa_path, str) and msa_path not in ["empty", ""] and Path(msa_path).exists():
                        
                        if not hasattr(self, '_msa_load_first_print'):
                            print(f"   [MSA] Loading MSA from: {msa_path} for chain_id={chain_id}")
                            if max_msa_seqs is None:
                                print(f"   [MSA] No parse-time seq cap (full file after de-dup)")
                            else:
                                print(f"   [MSA] Limiting parse to {max_msa_seqs} sequences")
                            self._msa_load_first_print = True
                        
                        try:
                            
                            msa_path_obj = Path(msa_path)
                            if msa_path_obj.suffix.lower() == '.csv':
                                from boltz.data.parse.csv import parse_csv
                                msa_obj = parse_csv(msa_path_obj, max_seqs=max_msa_seqs)
                            else:
                                
                                msa_obj = parse_a3m(msa_path_obj, taxonomy=None, max_seqs=max_msa_seqs)
                            msa_dict[chain_id] = msa_obj
                            
                            if not hasattr(self, '_msa_loaded_first_print'):
                                if hasattr(msa_obj, 'sequences') and msa_obj.sequences is not None:
                                    num_seqs = len(msa_obj.sequences)
                                    print(f"   [MSA] Chain {chain_id}: {num_seqs} sequences loaded")
                                self._msa_loaded_first_print = True
                        except Exception as e:
                            print(f"   [WARN] Failed to load MSA: {e}")
            
            
            if not hasattr(self, '_msa_summary_first_print'):
                if msa_dict:
                    print(f" [MSA] Loaded MSA for {len(msa_dict)} chain(s)")
                else:
                    print(f" [WARN] No MSAs loaded")
                self._msa_summary_first_print = True
            
            
            from boltz.data.types import Input
            input_data = Input(
                structure=target.structure,
                msa=msa_dict,  
                record=target.record,
                residue_constraints=target.residue_constraints,
                templates=target.templates,
                extra_mols=target.extra_mols or {},
            )
            
            # 5. Tokenize
            tokenized = self.tokenizer.tokenize(input_data)
            
            # 6. Featurize
            molecules = {}
            molecules.update(self.canonicals)
            if input_data.extra_mols:
                molecules.update(input_data.extra_mols)
            if getattr(self, "_bundled_mols", None):
                molecules.update(self._bundled_mols)
            
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(molecules.keys())
            if mol_names:
                from boltz.data.mol import load_molecules
                try:
                    loaded_mols = load_molecules(self.mol_dir, list(mol_names))
                    molecules.update(loaded_mols)
                    if not hasattr(self, '_ligand_loaded_first_print'):
                        print(f" [LIGAND] Loaded {len(loaded_mols)} molecule(s) from mol_dir: {sorted(mol_names)}")
                        self._ligand_loaded_first_print = True
                except (ValueError, KeyError) as e:
                    
                    print(f"\n [ERROR] Failed to load molecules from mol_dir: {sorted(mol_names)}")
                    print(f"   [ERROR] Error: {e}")
                    print(f"   [ERROR] mol_dir: {self.mol_dir}")
                    print(f" [ERROR] Tip: Ensure these CCD files exist in mol_dir (e.g., {sorted(mol_names)[0]}.pkl)")
                    print(f" [ERROR] Tip: Or use SMILES instead of CCD code:")
                    print(f"   [ERROR]      Example: --ligand 'C:smiles:O=P(O)(O)OP(=O)(O)OC[C@H]1O[C@H](n2cnc3c(N)ncnc32)[C@@H](O)[C@H]1O'")
                    raise ValueError(f"Failed to load molecules: {sorted(mol_names)}. Use SMILES or ensure CCD files exist in {self.mol_dir}") from e
            
            random = np.random.default_rng(42)
            
            
            inference_pocket_constraints = None
            inference_contact_constraints = None
            if hasattr(target, 'record') and target.record and hasattr(target.record, 'inference_options'):
                if target.record.inference_options:
                    inference_pocket_constraints = target.record.inference_options.pocket_constraints
                    inference_contact_constraints = target.record.inference_options.contact_constraints
                    if inference_contact_constraints:
                        if not hasattr(self, '_constraints_extracted_first_print'):
                            print(f" [CONSTRAINTS] extractto {len(inference_contact_constraints)} contactconstraint")
                            for token1, token2, max_dist, force in inference_contact_constraints:
                                print(f"      Contact: {token1} <-> {token2}, max_distance={max_dist:.1f}Å, force={force}")
                            self._constraints_extracted_first_print = True
            
            
            if hasattr(target, 'structure') and hasattr(target.structure, 'bonds'):
                bonds = target.structure.bonds
                if bonds is not None and len(bonds) > 0:
                    if not hasattr(self, '_bond_constraints_printed'):
                        print(f" [CONSTRAINTS] to {len(bonds)} bonds(andconstraint)")
                        
                        if 'chain_1' in bonds.dtype.names and 'chain_2' in bonds.dtype.names:
                            cross_chain_bonds_in_struct = []
                            for bond in bonds:
                                if bond['chain_1'] != bond['chain_2']:
                                    cross_chain_bonds_in_struct.append((
                                        bond['chain_1'], bond['chain_2'],
                                        bond['res_1'], bond['res_2'],
                                        bond['atom_1'], bond['atom_2']
                                    ))
                            if cross_chain_bonds_in_struct:
                                print(f" [CONSTRAINTS] inStructureinto {len(cross_chain_bonds_in_struct)} chainbonds(bondconstraint)")
                                for c1, c2, r1, r2, a1, a2 in cross_chain_bonds_in_struct[:3]:
                                    print(f"      Chain {c1} Res {r1} Atom {a1} <-> Chain {c2} Res {r2} Atom {a2}")
                            else:
                                print(f" [CONSTRAINTS] inStructureinnottochainbonds")
                        
                        
                        if hasattr(tokenized, 'bonds') and tokenized.bonds is not None:
                            token_bonds_count = len(tokenized.bonds)
                            print(f" [CONSTRAINTS] Tokenizedin {token_bonds_count} token_bonds")
                            
                            if token_bonds_count > 0 and 'token_1' in tokenized.bonds.dtype.names:
                                print(f" [CONSTRAINTS] 5token_bonds: {tokenized.bonds[:min(5, token_bonds_count)]}")
                                
                                cross_chain_bonds = []
                                if hasattr(tokenized, 'tokens') and tokenized.tokens is not None:
                                    
                                    token_chains = {}
                                    if 'asym_id' in tokenized.tokens.dtype.names:
                                        for i in range(len(tokenized.tokens)):
                                            token_chains[i] = tokenized.tokens[i]['asym_id']
                                    
                                    
                                    for bond in tokenized.bonds:
                                        t1 = bond['token_1']
                                        t2 = bond['token_2']
                                        if t1 in token_chains and t2 in token_chains:
                                            if token_chains[t1] != token_chains[t2]:
                                                cross_chain_bonds.append((t1, t2, token_chains[t1], token_chains[t2]))
                                
                                if cross_chain_bonds:
                                    print(f" [CONSTRAINTS] inTokenizedinto {len(cross_chain_bonds)} chaintoken_bonds(bondconstraintalready)")
                                    for t1, t2, c1, c2 in cross_chain_bonds[:5]:
                                        print(f"      Token {t1} (chain {c1}) <-> Token {t2} (chain {c2})")
                                else:
                                    print(f" [CONSTRAINTS] inTokenizedinnottochaintoken_bonds(bondconstraintnotconvertastoken_bonds)")
                        self._bond_constraints_printed = True
            
            
            
            from dream_boltz2.io.quiet import suppress_all_boltz2_output
            
            featurizer_max_seqs = (
                max_msa_seqs
                if max_msa_seqs is not None
                else BOLTZ_FEATURIZER_DEFAULT_MAX_SEQS
            )
            with suppress_all_boltz2_output():
                feats = self.featurizer.process(
                    tokenized,
                    molecules=molecules,
                    random=random,
                    training=False,
                    max_atoms=None,
                    max_tokens=None,
                    max_seqs=featurizer_max_seqs,
                    pad_to_max_seqs=False,
                    single_sequence_prop=0.0,
                    compute_frames=True,
                    inference_pocket_constraints=inference_pocket_constraints,
                    inference_contact_constraints=inference_contact_constraints,  
                    compute_constraint_features=True,
                    override_method=None,
                    compute_affinity=False,
                )
            
            
            if not hasattr(self, '_features_generated_first_print'):
                print(
                    f"   [OK] Features generated "
                    f"(parse cap={max_msa_seqs}, featurizer max_seqs={featurizer_max_seqs})"
                )
                
                
                if 'msa' in feats:
                    msa_feat = feats['msa']
                    print(f"   [MSA] MSA feature shape: {msa_feat.shape}")
                    print(f"   [MSA] Number of MSA sequences: {msa_feat.shape[0]}")
                else:
                    print(f"   [WARN] 'msa' key not found in features")
                
                
                if 'residue_type' in feats:
                    num_tokens = feats['residue_type'].shape[-1] if feats['residue_type'].dim() > 0 else 0
                    print(f"   [DEBUG] residue_type shape: {feats['residue_type'].shape if isinstance(feats['residue_type'], torch.Tensor) else 'N/A'}")
                elif 'res_type' in feats:
                    num_tokens = feats['res_type'].shape[-1] if feats['res_type'].dim() > 0 else 0
                    print(f"   [DEBUG] res_type shape: {feats['res_type'].shape if isinstance(feats['res_type'], torch.Tensor) else 'N/A'}")
                
                
                if hasattr(target, 'record') and target.record and hasattr(target.record, 'chains'):
                    print(f"   [DEBUG] Number of chains in target.record: {len(target.record.chains)}")
                    for i, chain_info in enumerate(target.record.chains):
                        print(f"     Chain {i}: asym_id={chain_info.chain_id}, name={getattr(chain_info, 'name', 'N/A')}")
                
                
                if 'type_bonds' in feats:
                    type_bonds = feats['type_bonds']  # [L, L] long tensor
                    if isinstance(type_bonds, torch.Tensor):
                        
                        from collections import Counter
                        type_bonds_np = type_bonds.cpu().numpy()
                        unique_types, counts = np.unique(type_bonds_np[type_bonds_np > 0], return_counts=True)
                        
                        
                        type_bond_names = {0: 'NONE', 1: 'OTHER', 2: 'SINGLE', 3: 'DOUBLE', 4: 'TRIPLE', 5: 'AROMATIC', 6: 'COVALENT'}
                        type_bond_counter = Counter()
                        for ut, cnt in zip(unique_types, counts):
                            type_name = type_bond_names.get(int(ut), f'UNKNOWN({ut})')
                            type_bond_counter[type_name] = cnt
                        
                        if type_bond_counter:
                            print(f" [TYPE_BONDS DEBUG] type_bondsin:")
                            for type_name, count in type_bond_counter.most_common():
                                print(f"      {type_name}: {count} ")
                            if type_bond_counter.get('AROMATIC', 0) > 0:
                                print(f" [TYPE_BONDS DEBUG] to {type_bond_counter['AROMATIC']} AROMATIC(thisvs6)")
                                
                                if 'mol_type' in feats:
                                    mol_type = feats['mol_type']
                                    if mol_type.dim() > 1:
                                        mol_type = mol_type.squeeze(0)
                                    ligand_mask = (mol_type != 0).cpu().numpy()  # ligand tokens
                                    ligand_indices = np.where(ligand_mask)[0]
                                    if len(ligand_indices) > 0:
                                        ligand_type_bonds = type_bonds_np[np.ix_(ligand_indices, ligand_indices)]
                                        aromatic_in_ligand = np.sum(ligand_type_bonds == 5)
                                        print(f" [TYPE_BONDS DEBUG] LigandAROMATIC: {aromatic_in_ligand} (: 12,asvs)")
                                        if aromatic_in_ligand == 12:
                                            print(f" [TYPE_BONDS DEBUG] LigandAROMATIC")
                                        else:
                                            print(f" [TYPE_BONDS DEBUG] LigandAROMATICnotmatch！")
                            else:
                                print(f" [TYPE_BONDS DEBUG] warning:type_bondsinnottoAROMATIC！")
                                print(f" thisligandnot()")
                
                self._features_generated_first_print = True
            
            
            
            feats_batched = {}
            for key, value in feats.items():
                if isinstance(value, torch.Tensor):
                    
                    value_batched = value.unsqueeze(0).to(self.device)
                    
                    
                    if key == 'template_mask' and value_batched.dim() == 3:
                        value_batched = value_batched.unsqueeze(-1)  # [B, T, L] -> [B, T, L, 1]
                    
                    feats_batched[key] = value_batched
                else:
                    feats_batched[key] = value
            
            return feats_batched
            
        finally:
            
            if temp_yaml.exists():
                temp_yaml.unlink()
    
    # ========================================================================
    
    # ========================================================================
    
    def prepare_complex_features(
        self,
        receptor_sequence: str,  
        binder_length: int,      
        binder_cyclic: bool = False,  
        receptor_coords: Optional[torch.Tensor] = None,  
        receptor_modifications: Optional[Dict[int, str]] = None,  
        receptor_msa_file: Optional[str] = None,  
        binder_msa_file: Optional[str] = None,  
        max_msa_seqs: Optional[int] = None,
        binder_template: Optional[str] = None,  
        binder_chains: Optional[List[Tuple[str, str]]] = None,  
        contact_constraints: Optional[List[Tuple[int, int, float]]] = None,  
        ligands: Optional[List[Dict]] = None,  
        template_file: Optional[str] = None,  
        template_chain_id: Optional[str] = None,  
        template_id: Optional[str] = None,  
        template_force: bool = False,  
        template_threshold: float = 3.0,  
        binder_modifications: Optional[List[Dict]] = None,  
        yaml_constraints: Optional[List] = None,  
        receptor_chain_id: Optional[str] = None,  
        binder_chain_id: Optional[str] = None,  
    ) -> Tuple[Dict, Dict]:
        """prepare complex features."""
        
        clean_receptor_seq, modified_mask = self._process_receptor_modifications(
            receptor_sequence,
            receptor_modifications
        )
        
        self.receptor_length = len(clean_receptor_seq)
        
        
        if binder_chains is not None and len(binder_chains) > 0:
            
            total_binder_length = sum(len(seq) for _, seq in binder_chains)
            self.binder_length = total_binder_length
            self.binder_slice = slice(self.receptor_length, self.receptor_length + total_binder_length)
            print(f"[FeaturePrep] chainbindermode: {len(binder_chains)} chain,length {total_binder_length}")
            for chain_id, chain_seq in binder_chains:
                print(f"  Chain {chain_id}: {len(chain_seq)} residue")
            binder_template_seq = None  
            
            full_sequence = clean_receptor_seq + ''.join(seq for _, seq in binder_chains)
        else:
            
            self.binder_length = binder_length
            self.binder_slice = slice(self.receptor_length, self.receptor_length + binder_length)
            
            
            
            
            if binder_template is None or (isinstance(binder_template, str) and len(binder_template.strip()) == 0):
                binder_template_seq = SUPER_RESIDUE * binder_length
                print(f"[FeaturePrep] usedefaultbindertemplate: {SUPER_RESIDUE} (notresidue,)")
            else:
                
                if len(binder_template) < binder_length:
                    binder_template_seq = (binder_template * ((binder_length // len(binder_template)) + 1))[:binder_length]
                elif len(binder_template) > binder_length:
                    binder_template_seq = binder_template[:binder_length]
                else:
                    binder_template_seq = binder_template
                print(f"[FeaturePrep] usebindertemplate: {binder_template_seq[:20]}..." if len(binder_template_seq) > 20 else f"[FeaturePrep] usebindertemplate: {binder_template_seq}")
            full_sequence = clean_receptor_seq + binder_template_seq
        
        
        self._receptor_sequence = clean_receptor_seq
        self._receptor_msa_file = receptor_msa_file
        self._max_msa_seqs = max_msa_seqs
        self._binder_cyclic = binder_cyclic
        self._ligands = ligands  
        self._yaml_constraints = yaml_constraints  
        
        
        
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'ccd') or not hasattr(self, 'mol_dir'):
            self._init_boltz_components()
        
        
        if receptor_msa_file is not None:
            print(f"[FeaturePrep] Using MSA file: {receptor_msa_file}")
            if max_msa_seqs is None:
                print(
                    f"[FeaturePrep] MSA parse: no row cap; "
                    f"featurizer max_seqs={BOLTZ_FEATURIZER_DEFAULT_MAX_SEQS} (Boltz default cap)"
                )
            else:
                print(f"[FeaturePrep] MSA parse limit: {max_msa_seqs} sequences")
        
        
        
        feats = self._generate_complex_features_with_msa(
            receptor_sequence=clean_receptor_seq,
            binder_sequence=binder_template_seq if (binder_chains is None or len(binder_chains) == 0) else None,  
            binder_chains=binder_chains,  
            receptor_msa_file=receptor_msa_file,  
            binder_msa_file=binder_msa_file,  
            binder_cyclic=binder_cyclic,
            binder_modifications=binder_modifications,  
            ligands=ligands,  
            max_msa_seqs=max_msa_seqs,
            contact_constraints=contact_constraints,  
            template_file=template_file,  
            template_chain_id=template_chain_id,  
            template_id=template_id,  
            template_force=template_force,  
            template_threshold=template_threshold,  
            yaml_constraints=yaml_constraints,  
            receptor_chain_id=receptor_chain_id,  
            binder_chain_id=binder_chain_id,  
        )
        
        
        
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                feats[k] = v.detach()
        
        
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                feats[k] = v.to(self.device)
        
        
        if modified_mask is not None:
            
            full_modified = torch.cat([
                modified_mask,
                torch.zeros(self.binder_length, dtype=torch.long, device=self.device)
            ], dim=0).unsqueeze(0)
            feats['modified'] = full_modified
        
        
        self._hard_feats = feats
        
        metadata = {
            'receptor_length': self.receptor_length,
            'target_length': self.receptor_length,  
            'binder_length': self.binder_length,
            'binder_slice': self.binder_slice,
            'receptor_slice': slice(0, self.receptor_length),  
            'binder_cyclic': binder_cyclic,  
            'mode': 'complex',
            'template': full_sequence,
            'ligands': ligands if ligands else [],  
        }
        
        
        if ligands:
            print(f"  [DEBUG] metadata['ligands'] = {metadata['ligands']}")
        else:
            print(f" [DEBUG] metadata['ligands'] = [] (ligandsargumentasNoneorempty)")
        
        return feats, metadata
    
    # ========================================================================
    
    # ========================================================================
    
    def update_sequence(
        self,
        feats: Dict,
        soft_sequence: torch.Tensor,  
        metadata: Dict,
        vocabulary=None  
    ) -> Dict:
        """update sequence."""
        mode = metadata['mode']
        
        if mode == 'monomer':
            return self._update_monomer_sequence(feats, soft_sequence, metadata, vocabulary)
        elif mode == 'complex':
            return self._update_complex_sequence(feats, soft_sequence, metadata, vocabulary)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _update_monomer_sequence(
        self,
        feats: Dict,
        soft_sequence: torch.Tensor,
        metadata: Dict,
        vocabulary=None
    ) -> Dict:
        """ update monomer sequence."""
        
        if vocabulary is not None and vocabulary.num_tokens > 20:
            
            return self._update_sequence_with_vocab(feats, soft_sequence, metadata, vocabulary)
        
        
        
        soft_seq_33 = self._convert_to_boltz_format(soft_sequence)
        self._binder_soft_res_type = soft_seq_33  
        
        
        
        
        feats['profile'] = soft_seq_33.detach()  # [1, L, 33]
        
        
        
        return feats
    
    def _update_sequence_with_vocab(
        self,
        feats: Dict,
        soft_sequence: torch.Tensor,  # [1, L, num_tokens]
        metadata: Dict,
        vocabulary
    ) -> Dict:
        """ update sequence with vocab."""
        mode = metadata['mode']
        
        
        with torch.no_grad():
            if soft_sequence.dim() == 3:
                indices = torch.argmax(soft_sequence, dim=-1)[0]  # [L]
            else:
                indices = torch.argmax(soft_sequence, dim=-1)  # [L]
            tokens = [vocabulary.idx_to_token[idx.item()] for idx in indices]
        
        
        base_sequence = vocabulary.get_base_sequence(tokens)
        modifications = vocabulary.get_modifications(tokens)
        
        
        
        
        
        if mode == 'complex':
            
            feats_new = self._generate_complex_features_with_msa(
                receptor_sequence=self._receptor_sequence,
                binder_sequence=base_sequence,
                receptor_msa_file=self._receptor_msa_file,
                max_msa_seqs=self._max_msa_seqs,
                binder_cyclic=self._binder_cyclic,
                binder_modifications=modifications,  
                ligands=self._ligands,  
                yaml_constraints=self._yaml_constraints  
            )
            
            
            self._binder_soft_sequence_56 = soft_sequence  # [1, binder_L, K]
            self._binder_hard_idx = soft_sequence[0].argmax(dim=-1).detach()

            if getattr(self, "_hard_validation_mode", False):
                for attr in ("_binder_soft_res_type", "_binder_soft_sequence_56"):
                    if hasattr(self, attr):
                        delattr(self, attr)
                self._last_feats = feats_new
                return feats_new
            
            if self.enable_soft_atoms:
                
                from dream_boltz2.model.soft_chemistry import (
                    build_candidate_token_basis,
                    mix_restype_modified,
                )
                if (
                    self._candidate_basis is None
                    or self._candidate_basis.num_tokens != vocabulary.num_tokens
                ):
                    print("[SoftChemistry] encoding candidate a_k via AtomEncoder "
                          f"(K={vocabulary.num_tokens}) ...")
                    self._candidate_basis = build_candidate_token_basis(
                        self.model.input_embedder,
                        vocabulary.tokens,
                        device=soft_sequence.device,
                        frame_mode=getattr(self, "isolated_frame", "center"),
                    )
                    print(
                        f"[SoftChemistry] basis K={self._candidate_basis.num_tokens} "
                        f"a={tuple(self._candidate_basis.a_basis.shape)} "
                        f"mode={self.soft_basis_mode} frame={self.isolated_frame}"
                    )
                    for w in self._candidate_basis.warnings[:8]:
                        print(f"  [warn] {w}")
                r_soft, _m_soft = mix_restype_modified(
                    soft_sequence,
                    self._candidate_basis.restype_basis,
                    self._candidate_basis.modified_basis,
                )
                self._binder_soft_res_type = r_soft.unsqueeze(0)
                
                
                self._print_binder_msa_once(feats_new, metadata.get("binder_slice", self.binder_slice))
            else:
                
                soft_20 = self._map_extended_to_base(soft_sequence, vocabulary)  # [1, binder_L, 20]
                soft_33 = self._convert_to_boltz_format(soft_20)  # [1, binder_L, 33]
                self._binder_soft_res_type = soft_33

            self._last_feats = feats_new
            return feats_new
        else:
            
            raise NotImplementedError("Extended vocab for monomer mode not yet implemented")
    
    def _update_complex_sequence(
        self,
        feats: Dict,
        soft_sequence: torch.Tensor,  # [1, binder_L, 20]
        metadata: Dict,
        vocabulary=None
    ) -> Dict:
        """ update complex sequence."""
        
        if vocabulary is not None and vocabulary.num_tokens > 20:
            return self._update_sequence_with_vocab(feats, soft_sequence, metadata, vocabulary)
        
        
        
        soft_seq_33 = self._convert_to_boltz_format(soft_sequence)
        self._binder_soft_res_type = soft_seq_33
        
        
        
        
        binder_slice = metadata['binder_slice']
        feats['profile'][0, binder_slice, :] = soft_seq_33[0].detach()  # [binder_L, 33]
        
        
        
        return feats
    
    # ========================================================================
    
    # ========================================================================
    
    def apply_injection(
        self,
        s_inputs: torch.Tensor,  
        metadata: Dict,
        feats: Optional[Dict] = None,
    ) -> torch.Tensor:
        """apply injection."""
        if getattr(self, "_hard_validation_mode", False):
            return s_inputs

        if (
            self.enable_soft_atoms
            and self._candidate_basis is not None
            and hasattr(self, "_binder_soft_sequence_56")
            and self._binder_soft_sequence_56 is not None
        ):
            return self._apply_soft_chemistry_injection(s_inputs, metadata, feats)

        if metadata['mode'] != 'complex':
            
            
            
            
            if not hasattr(self.model, 'input_embedder'):
                raise AttributeError("Boltz2 model must have 'input_embedder' attribute")
            
            res_type_encoding = self.model.input_embedder.res_type_encoding
            
            
            binder_slice = metadata['binder_slice']  # slice(0, length)
            
            
            hard_res_type = self._hard_feats['res_type'][0, binder_slice]  # [length, 33]
            
            
            if not hasattr(self, '_binder_soft_res_type'):
                raise RuntimeError("Must call update_sequence before apply_injection")
            
            soft_res_type = self._binder_soft_res_type[0]  # [length, 33]
            
            
            soft_embed = self._compute_embedding(soft_res_type, res_type_encoding)
            
            
            hard_embed = self._compute_embedding(hard_res_type, res_type_encoding).detach()
            
            
            delta_base = soft_embed - hard_embed  
            
            
            s_final = s_inputs + delta_base
            
            
            
            
            return s_final
        
        
        binder_slice = metadata['binder_slice']
        
        
        if not hasattr(self.model, 'input_embedder'):
            raise AttributeError("Boltz2 model must have 'input_embedder' attribute")
        
        res_type_encoding = self.model.input_embedder.res_type_encoding
        
        
        binder_hard_res_type = self._hard_feats['res_type'][0, binder_slice]  # [binder_L, 33]
        
        
        if not hasattr(self, '_binder_soft_res_type'):
            raise RuntimeError("Must call update_sequence before apply_injection")
        
        binder_soft_res_type = self._binder_soft_res_type[0]  # [binder_L, 33]
        
        
        soft_embed = self._compute_embedding(binder_soft_res_type, res_type_encoding)
        
        
        hard_embed = self._compute_embedding(binder_hard_res_type, res_type_encoding).detach()
        
        
        delta_base = soft_embed - hard_embed  
        
        
        delta_mod = torch.zeros_like(delta_base)  # [binder_L, 384]
        
        if hasattr(self, '_binder_soft_sequence_56') and self._binder_soft_sequence_56 is not None:
            
            soft_56 = self._binder_soft_sequence_56  # [1, binder_L, 56]
            prob_modified = soft_56[0, :, 20:].sum(dim=-1)  
            
            
            if hasattr(self.model, 'input_embedder'):
                if hasattr(self.model.input_embedder, 'modified_conditioning_init'):
                    mod_embedding = self.model.input_embedder.modified_conditioning_init
                    
                    # Soft modified embedding: (1-m) E_0 + m E_1
                    e0 = mod_embedding.weight[0]
                    e1 = mod_embedding.weight[1]
                    m = prob_modified.unsqueeze(-1)
                    soft_mod_embed = (1.0 - m) * e0 + m * e1
                    
                    # Hard modified embedding
                    hard_modified = self._hard_feats['modified'][0, binder_slice]  # [binder_L]
                    hard_mod_embed = mod_embedding(hard_modified).detach()  # [binder_L, 384]
                    
                    
                    delta_mod = soft_mod_embed - hard_mod_embed
                    
                    
                    if not hasattr(self, '_mod_channel_debug'):
                        print(f"[ModChannel] Double-channel injection enabled:")
                        print(f"  prob_modified range: [{prob_modified.min():.3f}, {prob_modified.max():.3f}]")
                        print(f"  delta_base norm: {delta_base.norm():.3f}")
                        print(f"  delta_mod norm: {delta_mod.norm():.3f}")
                        self._mod_channel_debug = True
        
        
        delta_total = delta_base + delta_mod
        
        
        
        
        
        s_final = torch.cat([
            s_inputs[0, :binder_slice.start],  
            s_inputs[0, binder_slice] + delta_total,  
            s_inputs[0, binder_slice.stop:],  
        ], dim=0).unsqueeze(0)
        
        
        
        
        
        return s_final

    def _apply_soft_chemistry_injection(
        self,
        s_inputs: torch.Tensor,
        metadata: Dict,
        feats: Optional[Dict] = None,
    ) -> torch.Tensor:
        """ apply soft chemistry injection."""
        from dream_boltz2.model.soft_chemistry import (
            compute_token_a,
            mix_candidate_chemistry,
            mix_delta_chemistry,
        )

        binder_slice = metadata["binder_slice"]
        embedder = self.model.input_embedder
        feats = feats if feats is not None else self._last_feats
        if feats is None:
            raise RuntimeError("soft chemistry injection needs feats from the current hard CCD")

        probs = self._binder_soft_sequence_56  # [1, Lb, K]
        basis = self._candidate_basis
        mod_emb = getattr(embedder, "modified_conditioning_init", None)
        use_mod = bool(getattr(embedder, "add_modified_flag", False) and mod_emb is not None)

        if self.soft_basis_mode == "delta":
            if getattr(self, "_binder_hard_idx", None) is not None:
                hard_idx = self._binder_hard_idx.to(probs.device)
            else:
                hard_idx = probs[0].argmax(dim=-1)
            delta = mix_delta_chemistry(
                probs,
                hard_idx,
                basis.a_basis,
                basis.restype_basis,
                basis.modified_basis,
                embedder.res_type_encoding,
                modified_embedding=mod_emb if use_mod else None,
            )
        else:
            e_soft = mix_candidate_chemistry(
                probs,
                basis.a_basis,
                basis.restype_basis,
                basis.modified_basis,
                embedder.res_type_encoding,
                modified_embedding=mod_emb if use_mod else None,
            )
            captured = getattr(embedder, "_captured_a", None)
            with torch.no_grad():
                if captured is not None and captured.shape[:2] == s_inputs.shape[:2]:
                    a_hard = captured
                else:
                    a_hard = compute_token_a(embedder, feats)
                r_hard = feats["res_type"][0, binder_slice].float()
                e_r_hard = embedder.res_type_encoding(r_hard)
                e_m_hard = torch.zeros_like(e_r_hard)
                if use_mod and "modified" in feats:
                    e_m_hard = mod_emb(feats["modified"][0, binder_slice])
                e_hard = a_hard[0, binder_slice] + e_r_hard + e_m_hard
            delta = e_soft - e_hard.detach()

        s_final = torch.cat(
            [
                s_inputs[0, :binder_slice.start],
                s_inputs[0, binder_slice] + delta,
                s_inputs[0, binder_slice.stop:],
            ],
            dim=0,
        ).unsqueeze(0)

        if not hasattr(self, "_soft_chem_debug"):
            print(
                f"[SoftChemistry] mode={self.soft_basis_mode} "
                f"injected; atom topology left hard; "
                f"delta norm={delta.norm().item():.4f}"
            )
            self._soft_chem_debug = True
        return s_final

    def _compute_embedding(
        self,
        res_type: torch.Tensor,  # [L, 33]
        embedding_layer: nn.Module  
    ) -> torch.Tensor:
        """ compute embedding."""
        
        if res_type.dtype != torch.float32:
            res_type = res_type.float()
        
        if isinstance(embedding_layer, nn.Linear):
            
            return embedding_layer(res_type)
        elif isinstance(embedding_layer, nn.Embedding):
            
            return torch.matmul(res_type, embedding_layer.weight)
        else:
            raise TypeError(f"Unsupported embedding layer type: {type(embedding_layer)}")
    
    # ========================================================================
    
    # ========================================================================
    
    def _map_extended_to_base(
        self,
        soft_sequence: torch.Tensor,  # [B, L, 56]
        vocabulary
    ) -> torch.Tensor:  # [B, L, 20]
        """ map extended to base."""
        
        device = soft_sequence.device
        mapping_matrix = torch.zeros(vocabulary.num_tokens, 20, device=device)
        
        
        from dream_boltz2.sequence.alphabet import AA_TO_IDX
        
        for idx, token in enumerate(vocabulary.tokens):
            
            base_aa_1letter = vocabulary.token_to_letter.get(token, 'X')
            
            
            if base_aa_1letter in AA_TO_IDX:
                base_idx = AA_TO_IDX[base_aa_1letter]
                mapping_matrix[idx, base_idx] = 1.0
        
        
        # [B, L, 56] @ [56, 20] -> [B, L, 20]
        soft_20 = torch.matmul(soft_sequence, mapping_matrix)
        
        return soft_20
    
    def _convert_to_boltz_format(
        self,
        soft_sequence: torch.Tensor  # [B, L, 20]
    ) -> torch.Tensor:
        """ convert to boltz format."""
        B, L, num_tokens = soft_sequence.shape
        
        
        
        if num_tokens != 20:
            raise ValueError(
                f"_convert_to_boltz_format20input,to{num_tokens}."
                f"ifuse,this_update_sequence_with_vocab"
            )
        
        soft_seq_33 = torch.zeros(
            B, L, 33,
            dtype=soft_sequence.dtype,
            device=soft_sequence.device
        )
        
        
        soft_seq_33[:, :, 2:22] = soft_sequence
        
        return soft_seq_33
    
    def _detach_non_sequence_features(
        self,
        feats: Dict
    ) -> Dict:
        """ detach non sequence features."""
        detached_feats = {}
        
        
        
        
        gradient_fields = set()  
        
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                if k in gradient_fields:
                    detached_feats[k] = v  
                else:
                    detached_feats[k] = v.detach()  # Detach
            else:
                detached_feats[k] = v  
        
        return detached_feats
    
    def _process_receptor_modifications(
        self,
        receptor_sequence: str,
        modifications: Optional[Dict[int, str]] = None
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """ process receptor modifications."""
        if modifications is None or len(modifications) == 0:
            
            return receptor_sequence, None
        
        
        clean_sequence = list(receptor_sequence)
        modified_mask = torch.zeros(len(receptor_sequence), dtype=torch.long)
        
        for pos, mod_type in modifications.items():
            if pos < 0 or pos >= len(receptor_sequence):
                raise ValueError(f"position {pos} sequence [0, {len(receptor_sequence)-1}]")
            
            
            if mod_type in MODIFIED_RESIDUE_MAP:
                standard_aa = MODIFIED_RESIDUE_MAP[mod_type]
                clean_sequence[pos] = standard_aa
                modified_mask[pos] = 1
                
                print(f"[Modifications] position {pos}: {mod_type} -> {standard_aa} (modified=1)")
            else:
                
                warnings.warn(
                    f"not '{mod_type}' inposition {pos}.\n"
                    f"support:{list(MODIFIED_RESIDUE_MAP.keys())}\n"
                    f"keepresidue:{receptor_sequence[pos]}"
                )
        
        clean_sequence = ''.join(clean_sequence)
        
        return clean_sequence, modified_mask


# ============================================================================

# ============================================================================

class FeaturePreparator(FeaturePreparatorComplex):
    """FeaturePreparator."""
    
    def __init__(self, boltz_model, device: str = 'cuda'):
        super().__init__(boltz_model, device, mode='monomer')
    
    def prepare_base_features(
        self,
        length: int,
        is_cyclic: bool = False,
        constraints: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """prepare base features."""
        return self.prepare_monomer_features(length, is_cyclic, constraints)


# ============================================================================

# ============================================================================

"\nBinder support(Phase 2 )\n\n:\n1. Binder needoptimization,need\n2. modified_conditioning nn.Embedding(support,notsupport)\n3. need Sequence support modified_logits\n\n:\n\n1. SequenceRepresentation:\n ```python\n class SequenceRepresentation:\n def __init__(self, length, support_modifications=False):\n self.logits = nn.Parameter(torch.randn(1, length, 20))\n \n if support_modifications:\n self.modified_logits = nn.Parameter(torch.zeros(1, length))\n \n def sample(self, temperature=1.0):\n soft_aa = gumbel_softmax(self.logits, tau=temperature, hard=True)\n \n if hasattr(self, 'modified_logits'):\n modified_prob = torch.sigmoid(self.modified_logits)\n else:\n modified_prob = None\n \n return soft_aa, modified_prob\n ```\n\n2. modified embedding:\n ```python\n # get\n embed_0 = modified_conditioning.weight[0] # residue\n embed_1 = modified_conditioning.weight[1] # residue\n \n # ()\n soft_modified_embed = (\n (1 - modified_prob[:, None]) * embed_0 +\n modified_prob[:, None] * embed_1\n )\n ```\n\n3. in apply_injection in:\n ```python\n def apply_injection(self, s_inputs, metadata, modified_prob=None):\n #......\n \n if modified_prob is not None:\n # modified embedding\n soft_modified_embed = self._compute_soft_modified_embedding(modified_prob)\n \n # to s_inputs[binder]\n s_injected[binder_slice] += soft_modified_embed\n \n return s_injected\n ```\n\n:in\n:~2-3 \n:(notneed)\n\n:\n- ifneedoptimization Binder \n-: binder\n"


# ============================================================================

# ============================================================================

'\n CCD residuesupport(Phase 3 ,)\n\n:\n1. Token ID notexists(Boltz2 33 tokens)\n2. Embedding fixed(need Boltz2 weight)\n3. atomnotmatch(residueatom, , not)\n\n:\n\n A:toresidue()\n```python\ncustom_ccd_map = {\n \'XYZ\': \'A\', # residue -> residue\n...\n}\n\n# use modified \nfeats[\'res_type\'][pos] = one_hot(Ala)\nfeats[\'modified\'][pos] = 1\n```\n-:,can\n-:residue\n\n B: Token (need Boltz2)\n```python\n# tokens to 50 (33 + 17 )\nextended_tokens = [..., "XYZ",...]\n\n# res_type_encoding\noriginal_weight = res_type_encoding.weight # [33, 384]\nnew_weight = initialize_new_rows(...) # [17, 384]\nextended_weight = torch.cat([original_weight, new_weight]) # [50, 384]\n\n# need Boltz2\ntrain(boltz_model, custom_residue_data)\n```\n-:residue\n-:,need GPU \n\n C: res_type,useatom()\n- not\n\n:\n:(to)\n:\n\n:\n- inand\n- as\n- not\n\n:\n- Phase 1: notsupport\n- Phase 2: ifneed,use A( + modified)\n- Phase 3: \n'

