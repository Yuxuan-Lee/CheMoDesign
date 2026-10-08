"""PDB readers and writers used by design and waking."""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path



AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    
    'SEP': 'S',  
    'TPO': 'T',  
    'PTR': 'Y',  
    'MSE': 'M',  # Selenomethionine
    'HYP': 'P',  # Hydroxyproline
    
    'MLY': 'K',  # N6-methyllysine
    'MLZ': 'K',  # N6-methyllysine (alternative)
    'M3L': 'K',  # N6,N6,N6-trimethyllysine
    'ALY': 'K',  # N6-acetyllysine
    '2MR': 'R',  # N5-methylarginine
    'MME': 'M',  # N-methylmethionine
    'MHS': 'H',  # N-methylhistidine
    'DAL': 'A',  # D-alanine
    'DAR': 'R',  # D-arginine
    'DSN': 'N',  # D-asparagine
    'DSP': 'D',  # D-aspartic acid
    'DCY': 'C',  # D-cysteine
    'DGL': 'E',  # D-glutamic acid
    'DGN': 'Q',  # D-glutamine
    'DHI': 'H',  # D-histidine
    'DIL': 'I',  # D-isoleucine
    'DLE': 'L',  # D-leucine
    'DLY': 'K',  # D-lysine
    'MED': 'M',  # D-methionine
    'DPN': 'F',  # D-phenylalanine
    'DPR': 'P',  # D-proline
    'DSE': 'S',  # D-serine
    'DTH': 'T',  # D-threonine
    'DTR': 'W',  # D-tryptophan
    'DTY': 'Y',  # D-tyrosine
    'DVA': 'V',  # D-valine
    'AIB': 'A',  # Alpha-aminoisobutyric acid
    'NLE': 'L',  # Norleucine
    'NVA': 'V',  # Norvaline
    'ORN': 'K',  # Ornithine
    'CIR': 'R',  # Citrulline
    'SAR': 'G',  # Sarcosine
    'SEC': 'C',  # Selenocysteine
}


def parse_pdb_chain(
    pdb_file: str,
    chain_id: str = 'B',
    verbose: bool = True
) -> Tuple[str, torch.Tensor, Dict[int, int]]:
    """parse pdb chain."""
    try:
        from Bio.PDB import PDBParser, MMCIFParser
    except ImportError:
        raise ImportError(
            "BioPython is required for PDB parsing. "
            "Install it with: pip install biopython"
        )
    
    if verbose:
        print(f"[PDB] Parsing PDB file: {pdb_file}")
        print(f"[PDB] Target chain: {chain_id}")
    
    
    if pdb_file.endswith('.cif') or pdb_file.endswith('.mmcif'):
        parser = MMCIFParser(QUIET=True)
        if verbose:
            print(f"[PDB] Using MMCIFParser for .cif file")
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure('receptor', pdb_file)
    
    
    chain = None
    for model in structure:
        for c in model:
            if c.id == chain_id:
                chain = c
                break
        if chain:
            break
    
    if not chain:
        raise ValueError(f"Chain {chain_id} not found in PDB file")
    
    
    
    sequence = []
    ca_coords = []
    residue_map = {}  
    
    
    try:
        from Bio.PDB import protein_letters_3to1
        
        bio_aa_map = protein_letters_3to1
    except ImportError:
        bio_aa_map = {}
    
    for idx, residue in enumerate(chain.get_residues()):
        if residue.id[0] == ' ':  
            
            res_name = residue.resname.strip().upper()  
            
            
            has_ca = False
            if 'CA' in residue:
                ca = residue['CA']
                ca_coords.append(ca.coord)
                has_ca = True
            else:
                
                ca_coords.append([0.0, 0.0, 0.0])
                if verbose:
                    print(f"   [WARN] Residue {res_name}{residue.id[1]} has no CA atom")
            
            
            aa_letter = None
            
            
            if res_name in AA_MAP:
                aa_letter = AA_MAP[res_name]
            
            elif res_name in bio_aa_map:
                aa_letter = bio_aa_map[res_name]
                
                if aa_letter not in 'ARNDCQEGHILKMFPSTWYV':
                    if verbose and idx < 3:
                        print(f"   [INFO] BioPython mapped {res_name} to {aa_letter} (non-standard), using 'X'")
                    aa_letter = 'X'
            
            elif res_name == 'UNK' or res_name == 'XXX':
                aa_letter = 'X'
                if verbose and idx < 3:
                    print(f"   [INFO] UNK residue at position {residue.id[1]}, using 'X'")
            
            else:
                aa_letter = 'X'
                if verbose and idx < 3:
                    print(f"   [WARN] Unknown residue type: {res_name} at position {residue.id[1]}, using 'X'")
            
            sequence.append(aa_letter)
            
            
            if has_ca:
                pdb_num = residue.id[1]
                residue_map[pdb_num] = idx
    
    coords = torch.tensor(np.array(ca_coords), dtype=torch.float32)
    sequence_str = ''.join(sequence)
    
    
    if len(sequence_str) != len(ca_coords):
        if len(ca_coords) > len(sequence_str):
            
            sequence_str = sequence_str + 'X' * (len(ca_coords) - len(sequence_str))
            if verbose:
                print(f"   [INFO] Added {len(ca_coords) - len(sequence_str)} 'X' to match coordinate count")
        else:
            
            sequence_str = sequence_str[:len(ca_coords)]
            if verbose:
                print(f"   [WARN] Truncated sequence to match coordinate count")
    
    if verbose:
        print(f"[PDB] [OK] Extracted chain {chain_id}:")
        print(f"   Length: {len(sequence_str)} residues")
        print(f"   Sequence: {sequence_str[:60]}{'...' if len(sequence_str) > 60 else ''}")
        print(f"   Coords shape: {coords.shape}")
    
    return sequence_str, coords, residue_map


def parse_pdb_complex(
    pdb_file: str,
    receptor_chain: str = 'B',
    binder_chain: str = 'A',
    verbose: bool = True
) -> Dict:
    """parse pdb complex."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Parsing PDB Complex: {pdb_file}")
        print(f"{'='*80}\n")
    
    
    receptor_seq, receptor_coords, receptor_map = parse_pdb_chain(
        pdb_file, receptor_chain, verbose=verbose
    )
    
    
    try:
        binder_seq, binder_coords, binder_map = parse_pdb_chain(
            pdb_file, binder_chain, verbose=verbose
        )
    except ValueError:
        
        binder_seq = None
        binder_coords = None
        binder_map = None
        if verbose:
            print(f"[PDB] No binder chain ({binder_chain}) found")
    
    complex_info = {
        'receptor_sequence': receptor_seq,
        'receptor_coords': receptor_coords,
        'receptor_residue_map': receptor_map,
        'receptor_length': len(receptor_seq),
        'receptor_chain': receptor_chain,
        'binder_sequence': binder_seq,
        'binder_coords': binder_coords,
        'binder_residue_map': binder_map,
        'binder_length': len(binder_seq) if binder_seq else 0,
        'binder_chain': binder_chain,
        'pdb_file': pdb_file,
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Complex Info Summary:")
        print(f"  Receptor ({receptor_chain}): {complex_info['receptor_length']} residues")
        if binder_seq:
            print(f"  Binder ({binder_chain}): {complex_info['binder_length']} residues")
        print(f"{'='*80}\n")
    
    return complex_info


def map_hotspot_residues(
    hotspot_pdb_nums: list,
    residue_map: Dict[int, int],
    verbose: bool = True
) -> list:
    """map hotspot residues."""
    hotspot_indices = []
    
    if verbose:
        print(f"[Hotspot] Mapping hotspot residues...")
    
    for pdb_num in hotspot_pdb_nums:
        if pdb_num in residue_map:
            idx = residue_map[pdb_num]
            hotspot_indices.append(idx)
            if verbose:
                print(f"   PDB#{pdb_num:3d} -> Index {idx:3d}")
        else:
            if verbose:
                print(f"   [WARNING] PDB#{pdb_num} not found in chain!")
    
    if not hotspot_indices:
        print(f"   [ERROR] ERROR: No valid hotspot residues found!")
    
    return hotspot_indices


def extract_disto_coords_from_pdb(
    pdb_file: str,
    chain_ids: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """extract disto coords from pdb."""
    try:
        from Bio.PDB import PDBParser, MMCIFParser
    except ImportError:
        raise ImportError(
            "BioPython is required for PDB parsing. "
            "Install it with: pip install biopython"
        )
    
    
    if pdb_file.endswith('.cif') or pdb_file.endswith('.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    
    try:
        from boltz.data import const
        res_to_disto_atom = const.res_to_disto_atom
        canonical_tokens = const.canonical_tokens
    except ImportError:
        
        if verbose:
            print("  [WARN] boltz module not found, using hardcoded mappings")
        
        
        res_to_disto_atom = {
            "UNK": "CB", "ALA": "CB", "ARG": "CB", "ASN": "CB", "ASP": "CB",
            "CYS": "CB", "GLN": "CB", "GLU": "CB", "GLY": "CA",  
            "HIS": "CB", "ILE": "CB", "LEU": "CB", "LYS": "CB", "MET": "CB",
            "PHE": "CB", "PRO": "CB", "SER": "CB", "THR": "CB", "TRP": "CB",
            "TYR": "CB", "VAL": "CB",
            "A": "C4", "G": "C4", "C": "C2", "U": "C2", "N": "C1'",
            "DA": "C4", "DG": "C4", "DC": "C2", "DT": "C2", "DN": "C1'"
        }
        canonical_tokens = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL", "UNK"
        ]
    
    if verbose:
        print(f"[Distogram] Extracting disto coordinates from: {pdb_file}")
    
    # parser already defined above based on file extension
    structure = parser.get_structure('structure', pdb_file)
    
    
    
    chain_data = {}  # {chain_id: [(coord, res_name, res_type), ...]}
    
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_ids is not None and chain_id not in chain_ids:
                continue
            
            chain_residues = []
            
            if verbose:
                print(f"  Processing chain {chain_id}...")
            
            for residue in chain.get_residues():
                res_name = residue.resname.strip()
                hetflag = residue.id[0]  # ' ' for standard, 'W' for water, 'H_' for hetero
                
                
                if hetflag == ' ':
                    
                    if res_name in canonical_tokens or res_name == 'UNK':
                        res_type = 'protein'
                    elif res_name in ['A', 'G', 'C', 'U', 'N']:
                        res_type = 'rna'
                    elif res_name in ['DA', 'DG', 'DC', 'DT', 'DN']:
                        res_type = 'dna'
                    elif 'CA' in residue:
                        
                        res_type = 'protein'
                        if verbose:
                            print(f"    [INFO] non-canonical {res_name} kept as protein (CA)")
                    else:
                        if verbose:
                            print(f"    [WARN] Unknown standard residue: {res_name}, skipping")
                        continue
                else:
                    # HETATM(ligand)
                    res_type = 'ligand'
                
                
                if res_type == 'ligand':
                    
                    atom_coords = []
                    for atom in residue:
                        atom_coords.append(atom.coord)
                    
                    if len(atom_coords) == 0:
                        if verbose:
                            print(f"    [WARN] Ligand {res_name} has no atoms, skipping")
                        continue
                    
                    
                    center_coord = np.mean(atom_coords, axis=0)
                    chain_residues.append((center_coord, res_name, res_type))
                    
                else:
                    
                    if res_name in res_to_disto_atom:
                        atom_name = res_to_disto_atom[res_name]
                    else:
                        
                        if res_type == 'protein':
                            atom_name = 'CA'  
                        elif res_type in ['dna', 'rna']:
                            atom_name = 'C1\''  
                        else:
                            atom_name = 'CA'
                        
                        if verbose:
                            print(f"    [WARN] {res_name} not in res_to_disto_atom, using {atom_name}")
                    
                    
                    if atom_name in residue:
                        atom = residue[atom_name]
                        chain_residues.append((atom.coord, res_name, res_type))
                    else:
                        
                        if res_type == 'protein' and 'CA' in residue:
                            if verbose:
                                print(f"    [WARN] {res_name} missing {atom_name}, using CA as fallback")
                            atom = residue['CA']
                            chain_residues.append((atom.coord, res_name, res_type))
                        else:
                            if verbose:
                                print(f"    [WARN] {res_name} missing {atom_name} and no CA, skipping")
            
            if len(chain_residues) > 0:
                chain_data[chain_id] = chain_residues
                if verbose:
                    print(f"  Collected {len(chain_residues)} residues from chain {chain_id}")
    
    
    disto_coords = []
    res_names = []
    res_types = []
    
    if chain_ids is not None:
        
        for chain_id in chain_ids:
            if chain_id in chain_data:
                if verbose:
                    print(f"  Extracting chain {chain_id} ({len(chain_data[chain_id])} residues)...")
                for coord, res_name, res_type in chain_data[chain_id]:
                    disto_coords.append(coord)
                    res_names.append(res_name)
                    res_types.append(res_type)
            else:
                if verbose:
                    print(f"  [WARN] Chain {chain_id} not found in PDB file, skipping")
    else:
        
        if verbose:
            print(f"  No chain_ids specified, using PDB file order...")
        for chain_id in sorted(chain_data.keys()):
            if verbose:
                print(f"  Extracting chain {chain_id} ({len(chain_data[chain_id])} residues)...")
            for coord, res_name, res_type in chain_data[chain_id]:
                disto_coords.append(coord)
                res_names.append(res_name)
                res_types.append(res_type)
    
    if len(disto_coords) == 0:
        raise ValueError(f"No valid residues found in PDB file: {pdb_file}")
    
    disto_coords = torch.tensor(np.array(disto_coords), dtype=torch.float32)
    
    if verbose:
        print(f"  [OK] Extracted {len(disto_coords)} disto coordinates")
        print(f"     Protein: {res_types.count('protein')}, DNA: {res_types.count('dna')}, "
              f"RNA: {res_types.count('rna')}, Ligand: {res_types.count('ligand')}")
    
    return disto_coords, res_names, res_types


def extract_distogram_from_pdb(
    pdb_file: str,
    chain_ids: Optional[List[str]] = None,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """extract distogram from pdb."""
    import torch.nn.functional as F
    
    
    disto_coords, res_names, res_types = extract_disto_coords_from_pdb(
        pdb_file, chain_ids, verbose=verbose
    )
    
    L = len(disto_coords)
    
    
    distances = torch.cdist(disto_coords, disto_coords)  # [L, L]
    
    
    boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
    
    
    
    distogram_bins = (distances.unsqueeze(-1) > boundaries).sum(dim=-1).long()  # [L, L]
    
    
    distogram = F.one_hot(distogram_bins, num_classes=num_bins).float()  # [L, L, num_bins]
    
    if verbose:
        print(f"  [OK] Distogram shape: {distogram.shape} (one-hot encoding)")
        
        try:
            print(f"     Distance range: {distances.min():.2f} - {distances.max():.2f} A")
            print(f"     Contact pairs (< 8 A): {(distances < 8.0).sum().item() - L}")  
        except UnicodeEncodeError:
            
            print(f"     Distance range: {distances.min():.2f} - {distances.max():.2f} Angstrom")
            print(f"     Contact pairs (< 8 Angstrom): {(distances < 8.0).sum().item() - L}")  
    
    return distogram, distances


def extract_backbone_atoms(
    coords: torch.Tensor,  # [1, N_atoms, 3]
    feats: Dict,
    metadata: Dict
) -> Dict[str, torch.Tensor]:
    """extract backbone atoms."""
    from boltz.data.const import protein_backbone_atom_index
    import numpy as np
    
    coords_np = coords[0].cpu().numpy()  # [N_atoms, 3]
    N_atoms = coords_np.shape[0]
    
    
    if 'ref_atom_name_chars' not in feats:
        raise ValueError("feats must contain 'ref_atom_name_chars'")
    
    atom_name_chars = feats['ref_atom_name_chars']
    if isinstance(atom_name_chars, torch.Tensor):
        atom_name_chars = atom_name_chars.cpu().numpy()
    
    
    if atom_name_chars.ndim == 4:
        atom_name_chars = np.argmax(atom_name_chars[0], axis=-1)  # [N_atoms, 4]
    elif atom_name_chars.ndim == 3:
        atom_name_chars = atom_name_chars[0]  # [N_atoms, 4]
    elif atom_name_chars.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected atom_name_chars shape: {atom_name_chars.shape}")
    
    
    atom_names = []
    for i in range(min(N_atoms, atom_name_chars.shape[0])):
        try:
            name_chars = []
            for j in range(atom_name_chars.shape[1]):
                char_val = int(atom_name_chars[i, j])
                ascii_val = char_val + 32
                if 32 <= ascii_val < 127:
                    name_chars.append(chr(ascii_val))
                else:
                    name_chars.append(' ')
            name_str = ''.join(name_chars).strip()
            atom_names.append(name_str)
        except Exception:
            atom_names.append('')
    
    
    if 'atom_to_token' not in feats:
        raise ValueError("feats must contain 'atom_to_token'")
    
    atom_to_token = feats['atom_to_token']
    if isinstance(atom_to_token, torch.Tensor):
        atom_to_token = atom_to_token.cpu().numpy()
    
    
    if atom_to_token.ndim == 3:
        atom_to_token = atom_to_token[0]
    if atom_to_token.ndim == 2:
        if atom_to_token.shape[1] == 1:
            atom_to_token = atom_to_token[:, 0]
        elif atom_to_token.shape[0] == 1:
            atom_to_token = atom_to_token[0]
        else:
            atom_to_token = np.argmax(atom_to_token, axis=1)
    
    
    receptor_length = metadata['receptor_length']
    binder_length = metadata['binder_length']
    L_total = receptor_length + binder_length
    
    
    backbone_coords = {
        'N': np.full((L_total, 3), np.nan, dtype=np.float32),
        'CA': np.full((L_total, 3), np.nan, dtype=np.float32),
        'C': np.full((L_total, 3), np.nan, dtype=np.float32),
        'O': np.full((L_total, 3), np.nan, dtype=np.float32),
    }
    
    
    for atom_idx in range(min(N_atoms, len(atom_to_token))):
        token_idx = int(atom_to_token[atom_idx])
        if token_idx >= L_total:
            continue
        
        atom_name = atom_names[atom_idx]
        if atom_name in backbone_coords:
            backbone_coords[atom_name][token_idx] = coords_np[atom_idx]
    
    
    result = {}
    for atom_name in ['N', 'CA', 'C', 'O']:
        result[atom_name] = torch.from_numpy(backbone_coords[atom_name]).float()
    
    return result


def extract_all_atoms(
    coords: torch.Tensor,  # [1, N_atoms, 3]
    feats: Dict,
    metadata: Dict
) -> Dict[int, Dict[str, torch.Tensor]]:
    """extract all atoms."""
    import numpy as np
    
    coords_np = coords[0].cpu().numpy()  # [N_atoms, 3]
    N_atoms = coords_np.shape[0]
    
    
    if 'ref_atom_name_chars' not in feats:
        raise ValueError("feats must contain 'ref_atom_name_chars'")
    
    atom_name_chars = feats['ref_atom_name_chars']
    if isinstance(atom_name_chars, torch.Tensor):
        atom_name_chars = atom_name_chars.cpu().numpy()
    
    
    if atom_name_chars.ndim == 4:
        atom_name_chars = np.argmax(atom_name_chars[0], axis=-1)  # [N_atoms, 4]
    elif atom_name_chars.ndim == 3:
        atom_name_chars = atom_name_chars[0]  # [N_atoms, 4]
    elif atom_name_chars.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected atom_name_chars shape: {atom_name_chars.shape}")
    
    
    atom_names = []
    for i in range(min(N_atoms, atom_name_chars.shape[0])):
        try:
            name_chars = []
            for j in range(atom_name_chars.shape[1]):
                char_val = int(atom_name_chars[i, j])
                ascii_val = char_val + 32
                if 32 <= ascii_val < 127:
                    name_chars.append(chr(ascii_val))
                else:
                    name_chars.append(' ')
            name_str = ''.join(name_chars).strip()
            atom_names.append(name_str)
        except Exception:
            atom_names.append('')
    
    
    if 'atom_to_token' not in feats:
        raise ValueError("feats must contain 'atom_to_token'")
    
    atom_to_token = feats['atom_to_token']
    if isinstance(atom_to_token, torch.Tensor):
        atom_to_token = atom_to_token.cpu().numpy()
    
    
    if atom_to_token.ndim == 3:
        atom_to_token = atom_to_token[0]
    if atom_to_token.ndim == 2:
        if atom_to_token.shape[1] == 1:
            atom_to_token = atom_to_token[:, 0]
        elif atom_to_token.shape[0] == 1:
            atom_to_token = atom_to_token[0]
        else:
            atom_to_token = np.argmax(atom_to_token, axis=1)
    
    
    receptor_length = metadata['receptor_length']
    binder_length = metadata['binder_length']
    L_total = receptor_length + binder_length
    
    
    if 'res_type' in feats:
        actual_L = feats['res_type'].shape[1] if feats['res_type'].ndim >= 2 else feats['res_type'].shape[0]
        if isinstance(actual_L, torch.Tensor):
            actual_L = actual_L.item() if actual_L.numel() == 1 else actual_L.shape[0]
    else:
        if len(atom_to_token) > 0:
            actual_L = int(atom_to_token.max()) + 1
        else:
            actual_L = L_total
    
    
    result = {}
    for token_idx in range(actual_L):
        result[token_idx] = {}
    
    
    for atom_idx in range(min(N_atoms, len(atom_to_token))):
        token_idx = int(atom_to_token[atom_idx])
        if token_idx >= actual_L:
            continue
        
        atom_name = atom_names[atom_idx]
        if atom_name:  
            result[token_idx][atom_name] = torch.from_numpy(coords_np[atom_idx]).float()
    
    return result


def write_coords_to_pdb(
    coords: torch.Tensor,  # [1, N_atoms, 3]
    feats: Dict,
    metadata: Dict,
    output_file: str,
    ca_coords: Optional[torch.Tensor] = None,  
    use_backbone: bool = True,  
    use_full_sidechain: bool = False,  
    cdr_regions: Optional[List[tuple]] = None,  
                                                
    fixed_binder_indices: Optional[List[int]] = None,  
                                                       
):
    """write coords to pdb."""
    from boltz.data.const import tokens, prot_token_to_letter
    import numpy as np

    fixed_set = set(fixed_binder_indices or [])

    # Helper: determine B-factor for a binder residue (local_idx is 0-based within binder)
    # LigandMPNN / batch_design_sequences: B-factor >= 0.99 -> fixed
    def _binder_bfactor(local_idx: int) -> float:
        if local_idx in fixed_set:
            return 100.0  # covalent / explicit fixed -> keep AA (e.g. Gly)
        if cdr_regions is None:
            return 0.0
        for cdr_start, cdr_end in cdr_regions:
            if cdr_start <= local_idx < cdr_end:
                return 0.0  # CDR -> designable
        return 100.0  # Framework -> fixed
    
    
    if 'res_type' in feats:
        res_type = feats['res_type'][0].cpu().numpy()  # [L, 33]
        
        sequence = []
        for res_onehot in res_type:
            res_idx = res_onehot.argmax()
            if res_idx < len(tokens):
                token_name = tokens[res_idx]
                if token_name in prot_token_to_letter:
                    sequence.append(prot_token_to_letter[token_name])
                else:
                    sequence.append('X')
            else:
                sequence.append('X')
    else:
        
        binder_length = metadata['binder_length']
        receptor_length = metadata['receptor_length']
        sequence = ['X'] * (receptor_length + binder_length)
    
    
    all_atoms = None
    if use_full_sidechain:
        try:
            all_atoms = extract_all_atoms(coords, feats, metadata)
            total_atoms = sum(len(atoms) for atoms in all_atoms.values())
            print(f" Extracted all atoms (including sidechains) for PDB output ({total_atoms} atoms found)")
        except Exception as e:
            print(f" Failed to extract all atoms: {e}, falling back to backbone only")
            import traceback
            traceback.print_exc()
            use_full_sidechain = False
            all_atoms = None
    
    
    backbone_atoms = None
    if use_backbone and not use_full_sidechain:
        try:
            backbone_atoms = extract_backbone_atoms(coords, feats, metadata)
            n_found = sum((~torch.isnan(backbone_atoms[atom_name]).any(dim=1)).sum().item() for atom_name in ['N', 'CA', 'C', 'O'])
            n_expected = (metadata['receptor_length'] + metadata['binder_length']) * 4
            print(f" Extracted backbone atoms for PDB output ({n_found}/{n_expected} atoms found)")
        except Exception as e:
            print(f" Failed to extract backbone atoms: {e}, falling back to CA only")
            import traceback
            traceback.print_exc()
            use_backbone = False
            backbone_atoms = None
    
    
    if ca_coords is None and not use_backbone:
        coords_np = coords[0].cpu().numpy()  # [N_atoms, 3]
        N_atoms = coords_np.shape[0]
        
        
        if 'token_to_center_atom' in feats:
            token_to_center_atom = feats['token_to_center_atom'][0].cpu().numpy()  # [L, N_atoms]
        elif 'token_to_rep_atom' in feats:
            token_to_center_atom = feats['token_to_rep_atom'][0].cpu().numpy()  # [L, N_atoms]
        else:
            raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")
        
        
        ca_coords_np = token_to_center_atom @ coords_np  # [L, 3]
    elif ca_coords is not None:
        ca_coords_np = ca_coords.cpu().numpy()  # [L_total, 3]
    else:
        ca_coords_np = None
    
    
    binder_slice = metadata['binder_slice']
    receptor_length = metadata['receptor_length']
    binder_length = metadata['binder_length']
    L_total = receptor_length + binder_length  
    
    
    receptor_chain_id = metadata.get('receptor_chain_id', 'B')  
    binder_chain_id = metadata.get('binder_chain_id', 'A')  
    
    
    binder_chains = metadata.get('binder_chains', None)  
    
    
    with open(output_file, 'w') as f:
        f.write(f"REMARK Generated from DREAM-boltz2\n")
        f.write(f"REMARK File: {output_file}\n")
        if binder_chains:
            f.write(f"REMARK Fab mode: {len(binder_chains)} binder chains\n")
        if use_full_sidechain:
            f.write(f"REMARK Contains all atoms (backbone + sidechains)\n")
        elif use_backbone:
            f.write(f"REMARK Contains all backbone atoms (N, CA, C, O)\n")
        else:
            f.write(f"REMARK Contains CA atoms only\n")
        f.write("\n")
        
        atom_serial = 1
        
        
        receptor_res_names = None
        if 'res_name_list' in metadata and 'receptor' in metadata['res_name_list']:
            receptor_res_names = metadata['res_name_list']['receptor']
            print(f" usemetadatainreceptor res_name_list(residue)")
        
        
        for token_idx in range(receptor_length):
            
            if receptor_res_names is not None and token_idx < len(receptor_res_names):
                res_name_3letter = receptor_res_names[token_idx]  
            else:
                
                res_token = sequence[token_idx] if token_idx < len(sequence) else 'X'
                res_name_3letter = {
                    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
                    'X': 'UNK'
                }.get(res_token, 'UNK')
            
            if use_full_sidechain and all_atoms is not None and token_idx in all_atoms:
                residue_atoms = all_atoms[token_idx]
                backbone_order = ['N', 'CA', 'C', 'O']
                
                for atom_name in backbone_order:
                    if atom_name in residue_atoms:
                        atom_coords = residue_atoms[atom_name].cpu().numpy()
                        x, y, z = atom_coords
                        element = atom_name[0] if atom_name != 'CA' else 'C'
                        f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {receptor_chain_id}{token_idx+1:4d}    "
                               f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")
                        atom_serial += 1
                
                sidechain_atoms = {k: v for k, v in residue_atoms.items() if k not in backbone_order}
                for atom_name in sorted(sidechain_atoms.keys()):
                    atom_coords = sidechain_atoms[atom_name].cpu().numpy()
                    x, y, z = atom_coords
                    element = atom_name[0] if atom_name[0].isalpha() else 'C'
                    if element not in ['C', 'N', 'O', 'S']:
                        element = 'C'
                    f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {receptor_chain_id}{token_idx+1:4d}    "
                           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")
                    atom_serial += 1
            elif use_backbone and backbone_atoms is not None:
                for atom_name in ['N', 'CA', 'C', 'O']:
                    atom_coords = backbone_atoms[atom_name][token_idx].cpu().numpy()
                    if not np.isnan(atom_coords).any():
                        x, y, z = atom_coords
                        element = atom_name[0] if atom_name != 'CA' else 'C'
                        f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {receptor_chain_id}{token_idx+1:4d}    "
                               f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")
                        atom_serial += 1
            else:
                if ca_coords_np is not None and token_idx < ca_coords_np.shape[0]:
                    x, y, z = ca_coords_np[token_idx]
                    f.write(f"ATOM  {atom_serial:5d}  CA {res_name_3letter:3s} {receptor_chain_id}{token_idx+1:4d}    "
                           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
                    atom_serial += 1
        
        
        
        if binder_chains:
            
            for chain_id, chain_slice in binder_chains:
                
                chain_start = chain_slice.start
                chain_stop = chain_slice.stop
                chain_length = chain_stop - chain_start
                
                
                binder_res_names = None
                if 'res_name_list' in metadata and 'binder' in metadata['res_name_list']:
                    binder_res_names = metadata['res_name_list']['binder']
                
                
                for local_idx, token_idx in enumerate(range(chain_start, chain_stop)):
                    # binder-relative index for B-factor (cdr_regions is relative to binder start)
                    binder_rel_idx = token_idx - binder_slice.start
                    bfac = _binder_bfactor(binder_rel_idx)
                    
                    if binder_res_names is not None and local_idx < len(binder_res_names):
                        res_name_3letter = binder_res_names[local_idx]
                    else:
                        res_token = sequence[token_idx] if token_idx < len(sequence) else 'X'
                        res_name_3letter = {
                            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
                            'X': 'UNK'
                        }.get(res_token, 'UNK')

                    
                    if use_full_sidechain and all_atoms is not None and token_idx in all_atoms:
                        residue_atoms = all_atoms[token_idx]
                        backbone_order = ['N', 'CA', 'C', 'O']
                        for atom_name in backbone_order:
                            if atom_name in residue_atoms:
                                atom_coords = residue_atoms[atom_name].cpu().numpy()
                                x, y, z = atom_coords
                                element = atom_name[0] if atom_name != 'CA' else 'C'
                                f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {chain_id}{local_idx+1:4d}    "
                                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {element:>2s}\n")
                                atom_serial += 1
                        sidechain_atoms = {k: v for k, v in residue_atoms.items() if k not in backbone_order}
                        for atom_name in sorted(sidechain_atoms.keys()):
                            atom_coords = sidechain_atoms[atom_name].cpu().numpy()
                            x, y, z = atom_coords
                            element = atom_name[0] if atom_name[0].isalpha() else 'C'
                            if element not in ['C', 'N', 'O', 'S']:
                                element = 'C'
                            f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {chain_id}{local_idx+1:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {element:>2s}\n")
                            atom_serial += 1
                    elif use_backbone and backbone_atoms is not None:
                        for atom_name in ['N', 'CA', 'C', 'O']:
                            atom_coords = backbone_atoms[atom_name][token_idx].cpu().numpy()
                            if not np.isnan(atom_coords).any():
                                x, y, z = atom_coords
                                element = atom_name[0] if atom_name != 'CA' else 'C'
                                f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {chain_id}{local_idx+1:4d}    "
                                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {element:>2s}\n")
                                atom_serial += 1
                    else:
                        if ca_coords_np is not None and token_idx < ca_coords_np.shape[0]:
                            x, y, z = ca_coords_np[token_idx]
                            f.write(f"ATOM  {atom_serial:5d}  CA {res_name_3letter:3s} {chain_id}{local_idx+1:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}           C  \n")
                            atom_serial += 1
        else:
            
            binder_res_names = None
            if 'res_name_list' in metadata and 'binder' in metadata['res_name_list']:
                binder_res_names = metadata['res_name_list']['binder']
            
            for local_idx, token_idx in enumerate(range(binder_slice.start, binder_slice.stop)):
                if binder_res_names is not None and local_idx < len(binder_res_names):
                    res_name_3letter = binder_res_names[local_idx]
                else:
                    res_token = sequence[token_idx] if token_idx < len(sequence) else 'X'
                    res_name_3letter = {
                        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
                        'X': 'UNK'
                    }.get(res_token, 'UNK')
                
                bfac = _binder_bfactor(local_idx)
                if use_full_sidechain and all_atoms is not None and token_idx in all_atoms:
                    residue_atoms = all_atoms[token_idx]
                    backbone_order = ['N', 'CA', 'C', 'O']
                    for atom_name in backbone_order:
                        if atom_name in residue_atoms:
                            atom_coords = residue_atoms[atom_name].cpu().numpy()
                            x, y, z = atom_coords
                            element = atom_name[0] if atom_name != 'CA' else 'C'
                            f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {binder_chain_id}{local_idx+1:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {element:>2s}\n")
                            atom_serial += 1
                    sidechain_atoms = {k: v for k, v in residue_atoms.items() if k not in backbone_order}
                    for atom_name in sorted(sidechain_atoms.keys()):
                        atom_coords = sidechain_atoms[atom_name].cpu().numpy()
                        x, y, z = atom_coords
                        element = atom_name[0] if atom_name[0].isalpha() else 'C'
                        if element not in ['C', 'N', 'O', 'S']:
                            element = 'C'
                        f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {binder_chain_id}{local_idx+1:4d}    "
                               f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {element:>2s}\n")
                        atom_serial += 1
                elif use_backbone and backbone_atoms is not None:
                    for atom_name in ['N', 'CA', 'C', 'O']:
                        atom_coords = backbone_atoms[atom_name][token_idx].cpu().numpy()
                        if not np.isnan(atom_coords).any():
                            x, y, z = atom_coords
                            element = atom_name[0] if atom_name != 'CA' else 'C'
                            f.write(f"ATOM  {atom_serial:5d}  {atom_name:>3s} {res_name_3letter:3s} {binder_chain_id}{local_idx+1:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {element:>2s}\n")
                            atom_serial += 1
                else:
                    if ca_coords_np is not None and token_idx < ca_coords_np.shape[0]:
                        x, y, z = ca_coords_np[token_idx]
                        f.write(f"ATOM  {atom_serial:5d}  CA {res_name_3letter:3s} {binder_chain_id}{local_idx+1:4d}    "
                               f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}           C  \n")
                        atom_serial += 1
        
        
        
        ligand_tokens = []
        ligand_res_names = {}
        
        
        if 'mol_type' in feats:
            mol_type = feats['mol_type']
            if isinstance(mol_type, torch.Tensor):
                mol_type = mol_type[0].cpu().numpy() if mol_type.ndim >= 2 else mol_type.cpu().numpy()
            
            
            try:
                from boltz.data import const
                NONPOLYMER_ID = const.chain_type_ids.get("NONPOLYMER", 3)  
            except:
                NONPOLYMER_ID = 3  
            
            
            if 'res_name' in feats:
                res_name = feats['res_name']
                if isinstance(res_name, torch.Tensor):
                    res_name = res_name[0].cpu().numpy() if res_name.ndim >= 2 else res_name.cpu().numpy()
            else:
                res_name = None
            
            
            actual_L = L_total
            if 'token_pad_mask' in feats:
                token_pad_mask = feats['token_pad_mask']
                if isinstance(token_pad_mask, torch.Tensor):
                    token_pad_mask = token_pad_mask[0] if token_pad_mask.ndim >= 2 else token_pad_mask
                    actual_L = int(token_pad_mask.sum().item())  
            elif 'res_type' in feats:
                actual_L = feats['res_type'].shape[1] if feats['res_type'].ndim >= 2 else feats['res_type'].shape[0]
                if isinstance(actual_L, torch.Tensor):
                    actual_L = actual_L.item() if actual_L.numel() == 1 else actual_L.shape[0]
            
            
            
            mol_type_len = len(mol_type) if hasattr(mol_type, '__len__') else 0
            
            
            
            if mol_type_len >= actual_L:
                
                for token_idx in range(L_total, actual_L):
                    if token_idx < len(mol_type) and mol_type[token_idx] == NONPOLYMER_ID:
                        ligand_tokens.append(token_idx)
                        
                        if res_name is not None and token_idx < len(res_name):
                            ligand_res_names[token_idx] = res_name[token_idx]
            else:
                
                
                
                pass
        
        
        if not ligand_tokens and 'res_type' in feats:
            try:
                from boltz.data import const
                res_type = feats['res_type']
                if isinstance(res_type, torch.Tensor):
                    res_type = res_type[0].cpu().numpy() if res_type.ndim >= 2 else res_type.cpu().numpy()
                
                
                if 'res_name' in feats:
                    res_name = feats['res_name']
                    if isinstance(res_name, torch.Tensor):
                        res_name = res_name[0].cpu().numpy() if res_name.ndim >= 2 else res_name.cpu().numpy()
                else:
                    res_name = None
                
                
                standard_tokens = const.canonical_tokens[:20]  # ALA, ARG, ASN, ...
                
                
                actual_L = res_type.shape[0] if res_type.ndim == 1 else res_type.shape[1]
                for token_idx in range(L_total, actual_L):
                    
                    if res_type.ndim == 2:
                        res_idx = res_type[token_idx].argmax()
                    else:
                        res_idx = res_type[token_idx]
                    
                    
                    if res_idx >= len(const.tokens):
                        continue
                    
                    token_name = const.tokens[res_idx]
                    
                    if token_name not in standard_tokens and token_name not in ['UNK', 'X']:
                        ligand_tokens.append(token_idx)
                        if res_name is not None and token_idx < len(res_name):
                            ligand_res_names[token_idx] = res_name[token_idx]
                        else:
                            ligand_res_names[token_idx] = token_name
            except Exception as e:
                
                pass
        
        
        if not ligand_tokens:
            try:
                
                if 'atom_to_token' in feats:
                    atom_to_token = feats['atom_to_token']
                    if isinstance(atom_to_token, torch.Tensor):
                        atom_to_token = atom_to_token[0].cpu().numpy() if atom_to_token.ndim >= 2 else atom_to_token.cpu().numpy()
                    
                    
                    unique_tokens = np.unique(atom_to_token)
                    
                    max_token = int(unique_tokens.max()) if len(unique_tokens) > 0 else L_total
                    
                    
                    if max_token >= L_total:
                        
                        
                        if 'token_to_center_atom' in feats or 'token_to_rep_atom' in feats:
                            token_to_center_atom = feats.get('token_to_center_atom') or feats.get('token_to_rep_atom')
                            if isinstance(token_to_center_atom, torch.Tensor):
                                token_to_center_atom = token_to_center_atom[0].cpu().numpy() if token_to_center_atom.ndim >= 2 else token_to_center_atom.cpu().numpy()
                            
                            
                            token_to_center_atom_len = token_to_center_atom.shape[0] if token_to_center_atom.ndim >= 1 else 0
                            
                            
                            if token_to_center_atom_len > L_total:
                                
                                ligand_ccd_name = "LIG"  
                                if 'ligands' in metadata and isinstance(metadata['ligands'], list) and len(metadata['ligands']) > 0:
                                    ligand_info = metadata['ligands'][0]  
                                    if isinstance(ligand_info, dict):
                                        if 'ccd' in ligand_info and ligand_info['ccd']:
                                            ligand_ccd_name = ligand_info['ccd']
                                        elif 'smiles' in ligand_info and ligand_info['smiles']:
                                            
                                            ligand_ccd_name = "LIG"
                                
                                
                                for token_idx in range(L_total, token_to_center_atom_len):
                                    ligand_tokens.append(token_idx)
                                    
                                    if res_name is not None and token_idx < len(res_name):
                                        ligand_res_name = res_name[token_idx]
                                        
                                        if isinstance(ligand_res_name, (int, np.integer)):
                                            try:
                                                from boltz.data.const import tokens
                                                if ligand_res_name < len(tokens):
                                                    ligand_res_names[token_idx] = tokens[ligand_res_name]
                                                else:
                                                    ligand_res_names[token_idx] = ligand_ccd_name
                                            except:
                                                ligand_res_names[token_idx] = ligand_ccd_name
                                        else:
                                            ligand_res_names[token_idx] = str(ligand_res_name)
                                    else:
                                        
                                        ligand_res_names[token_idx] = ligand_ccd_name
            except Exception as e:
                
                pass
        
        
        if ligand_tokens:
            
            # for token_idx in ligand_tokens:
            #     ligand_name = ligand_res_names.get(token_idx, "LIG")
            #     print(f"     Token {token_idx}: {ligand_name}")
            pass
        else:
            
            # if not hasattr(write_coords_to_pdb, '_ligand_debug_printed'):
            
            
            #     write_coords_to_pdb._ligand_debug_printed = True
            pass
        
        
        if ligand_tokens:
            
            ligand_chain_id = 'C'  
            if 'ligands' in metadata and isinstance(metadata['ligands'], list) and len(metadata['ligands']) > 0:
                ligand_info = metadata['ligands'][0]  
                if isinstance(ligand_info, dict) and 'id' in ligand_info:
                    ligand_chain_id = ligand_info.get('id', 'C')
            
            
            if use_full_sidechain and all_atoms is not None:
                ligand_written = False
                for ligand_token_idx in ligand_tokens:
                    if ligand_token_idx in all_atoms:
                        ligand_atoms = all_atoms[ligand_token_idx]
                        
                        if ligand_token_idx in ligand_res_names:
                            ligand_res_name = ligand_res_names[ligand_token_idx]
                            if isinstance(ligand_res_name, (int, np.integer)):
                                
                                try:
                                    from boltz.data.const import tokens
                                    if ligand_res_name < len(tokens):
                                        ligand_name = tokens[ligand_res_name]
                                    else:
                                        ligand_name = "LIG"
                                except:
                                    ligand_name = "LIG"
                            else:
                                ligand_name = str(ligand_res_name)
                        elif res_name is not None and ligand_token_idx < len(res_name):
                            ligand_res_name = res_name[ligand_token_idx]
                            if isinstance(ligand_res_name, (int, np.integer)):
                                try:
                                    from boltz.data.const import tokens
                                    if ligand_res_name < len(tokens):
                                        ligand_name = tokens[ligand_res_name]
                                    else:
                                        ligand_name = "LIG"
                                except:
                                    ligand_name = "LIG"
                            else:
                                ligand_name = str(ligand_res_name)
                        else:
                            ligand_name = "LIG"
                        
                        
                        ligand_res_num = ligand_token_idx - L_total + 1
                        
                        
                        for atom_name in sorted(ligand_atoms.keys()):
                            atom_coords = ligand_atoms[atom_name].cpu().numpy()
                            x, y, z = atom_coords
                            
                            element = atom_name[0] if atom_name and atom_name[0].isalpha() else 'C'
                            if len(atom_name) > 1 and atom_name[1].islower():
                                element = atom_name[:2]  
                            
                            element = element[:2].upper()
                            
                            f.write(f"HETATM{atom_serial:5d} {atom_name:>4s} {ligand_name:>3s} {ligand_chain_id}{ligand_res_num:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")
                            atom_serial += 1
                        ligand_written = True
            else:
                ligand_written = False
            
            
            
            if not ligand_written:
                if 'token_to_center_atom' in feats or 'token_to_rep_atom' in feats:
                    coords_np = coords[0].cpu().numpy()  # [N_atoms, 3]
                    if 'token_to_center_atom' in feats:
                        token_to_center_atom = feats['token_to_center_atom']
                        if isinstance(token_to_center_atom, torch.Tensor):
                            token_to_center_atom = token_to_center_atom[0].cpu().numpy() if token_to_center_atom.ndim >= 2 else token_to_center_atom.cpu().numpy()
                    else:
                        token_to_center_atom = feats['token_to_rep_atom']
                        if isinstance(token_to_center_atom, torch.Tensor):
                            token_to_center_atom = token_to_center_atom[0].cpu().numpy() if token_to_center_atom.ndim >= 2 else token_to_center_atom.cpu().numpy()
                    
                    for ligand_token_idx in ligand_tokens:
                        if ligand_token_idx < len(token_to_center_atom):
                            
                            center_coord = token_to_center_atom[ligand_token_idx] @ coords_np  # [3]
                            x, y, z = center_coord
                            
                            
                            if ligand_token_idx in ligand_res_names:
                                ligand_res_name = ligand_res_names[ligand_token_idx]
                                if isinstance(ligand_res_name, (int, np.integer)):
                                    try:
                                        from boltz.data.const import tokens
                                        if ligand_res_name < len(tokens):
                                            ligand_name = tokens[ligand_res_name]
                                        else:
                                            ligand_name = ligand_ccd_name
                                    except:
                                        ligand_name = ligand_ccd_name
                                else:
                                    ligand_name = str(ligand_res_name)
                            elif res_name is not None and ligand_token_idx < len(res_name):
                                ligand_res_name = res_name[ligand_token_idx]
                                if isinstance(ligand_res_name, (int, np.integer)):
                                    try:
                                        from boltz.data.const import tokens
                                        if ligand_res_name < len(tokens):
                                            ligand_name = tokens[ligand_res_name]
                                        else:
                                            ligand_name = ligand_ccd_name
                                    except:
                                        ligand_name = ligand_ccd_name
                                else:
                                    ligand_name = str(ligand_res_name)
                            else:
                                ligand_name = "LIG"
                            
                            
                            ligand_res_num = ligand_token_idx - L_total + 1
                            
                            
                            f.write(f"HETATM{atom_serial:5d}  CA {ligand_name:>3s} {ligand_chain_id}{ligand_res_num:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
                            atom_serial += 1


if __name__ == "__main__":
    
    import sys
    from pathlib import Path
    
    
    test_pdb = Path(__file__).parent.parent.parent / "inverse_design" / "Z_and_B7H3.pdb"
    
    if test_pdb.exists():
        print("Testing PDB parsing...")
        
        
        info = parse_pdb_complex(str(test_pdb), 'B', 'A')
        
        
        hotspot_pdb_nums = [38, 55, 97]
        hotspot_indices = map_hotspot_residues(
            hotspot_pdb_nums,
            info['receptor_residue_map']
        )
        
        print(f"\n[OK] All tests passed!")
    else:
        print(f"[ERROR] Test PDB file not found: {test_pdb}")

