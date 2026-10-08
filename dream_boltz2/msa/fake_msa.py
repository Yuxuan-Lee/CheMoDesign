"""Synthetic paired MSA that places a coevolution-like signal on dreamed interface contacts."""

import random
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

from dream_boltz2.io.pdb import parse_pdb_chain


# ============================================================================

# ============================================================================


MIN_PAIRED_MSA_NUM_SEQS = 300

# ============================================================================

# ============================================================================


STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"




MUTABLE_AA = "ADEFGHIKLMNQRSTVWY"


POSITIVE_CHARGED = {"R", "K", "H"}  # Arg, Lys, His
NEGATIVE_CHARGED = {"D", "E"}       # Asp, Glu
NEUTRAL_POLAR = {"N", "Q", "S", "T", "Y"}  # Asn, Gln, Ser, Thr, Tyr


HYDROPHOBIC = {"A", "F", "I", "L", "M", "P", "V", "W"}  # Ala, Phe, Ile, Leu, Met, Pro, Val, Trp, Tyr
SPECIAL = {"G", "C"}  # Gly, Cys


CHARGE_COMPLEMENTARY = {
    "R": "E",  
    "K": "D",  
    "H": "D",  
    "E": "R",  
    "D": "K",  
}


HYDROPHOBIC_PAIRS = {
    "F": ["F", "I", "L", "V", "W", "Y"],  
    "I": ["F", "I", "L", "V", "W", "Y"],
    "L": ["F", "I", "L", "V", "W", "Y"],
    "V": ["F", "I", "L", "V", "W", "Y"],
    "W": ["F", "I", "L", "V", "W", "Y"],
    "Y": ["F", "I", "L", "V", "W", "Y"],
    "A": ["A", "V", "L", "I"],  
    "M": ["F", "I", "L", "V"],
    "P": ["F", "I", "L", "V"],
}

# ============================================================================

# ============================================================================



BLOSUM62_SIMILAR = {
    'A': ['S', 'T', 'V'],           
    'R': ['K', 'Q'],                
    'N': ['D', 'S', 'H'],           # Asn -> Asp, Ser, His
    'D': ['E', 'N'],                # Asp -> Glu, Asn
    'C': ['S'],                     # Cys -> Ser
    'Q': ['E', 'K', 'R'],           # Gln -> Glu, Lys, Arg
    'E': ['D', 'Q', 'K'],           # Glu -> Asp, Gln, Lys
    'G': ['A', 'S'],                # Gly -> Ala, Ser
    'H': ['N', 'Y'],                # His -> Asn, Tyr
    'I': ['L', 'V', 'M'],           
    'L': ['I', 'V', 'M', 'F'],      
    'K': ['R', 'Q', 'E'],           
    'M': ['L', 'I', 'V'],           # Met -> Leu, Ile, Val
    'F': ['Y', 'W', 'L'],           
    'P': ['A', 'S'],                
    'S': ['T', 'A', 'N'],           # Ser -> Thr, Ala, Asn
    'T': ['S', 'A'],                # Thr -> Ser, Ala
    'W': ['Y', 'F'],                
    'Y': ['F', 'H', 'W'],           
    'V': ['I', 'L', 'A', 'M'],      
}


def sample_conservative_mutation(original_aa: str, exclude_cp: bool = True) -> str:
    """sample conservative mutation."""
    original_aa = original_aa.upper()
    
    if original_aa not in BLOSUM62_SIMILAR:
        
        if exclude_cp:
            return random.choice([aa for aa in MUTABLE_AA if aa != original_aa])
        else:
            return random.choice([aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != original_aa])
    
    
    similar_aas = BLOSUM62_SIMILAR[original_aa]
    
    
    if exclude_cp:
        similar_aas = [aa for aa in similar_aas if aa not in {'C', 'P'}]
    
    
    similar_aas = [aa for aa in similar_aas if aa != original_aa]
    
    
    if not similar_aas:
        if exclude_cp:
            return random.choice([aa for aa in MUTABLE_AA if aa != original_aa])
        else:
            return random.choice([aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != original_aa])
    
    
    return random.choice(similar_aas)


def classify_aa(aa: str) -> str:
    """classify aa."""
    aa = aa.upper()
    if aa in POSITIVE_CHARGED:
        return "positive"
    elif aa in NEGATIVE_CHARGED:
        return "negative"
    elif aa in HYDROPHOBIC:
        return "hydrophobic"
    elif aa in NEUTRAL_POLAR:
        return "polar"
    elif aa in SPECIAL:
        return "special"
    else:
        return "unknown"


def get_complementary_aa(aa: str, mode: str = "complementary") -> str:
    """get complementary aa."""
    aa = aa.upper()
    
    if mode == "same":
        
        return aa
    
    elif mode == "complementary":
        
        aa_type = classify_aa(aa)
        
        if aa_type == "positive":
            
            if aa in CHARGE_COMPLEMENTARY:
                return CHARGE_COMPLEMENTARY[aa]
            else:
                
                return random.choice(list(HYDROPHOBIC))
        
        elif aa_type == "negative":
            
            if aa in CHARGE_COMPLEMENTARY:
                return CHARGE_COMPLEMENTARY[aa]
            else:
                return random.choice(list(POSITIVE_CHARGED))
        
        elif aa_type == "hydrophobic":
            
            if aa in HYDROPHOBIC_PAIRS:
                return random.choice(HYDROPHOBIC_PAIRS[aa])
            else:
                return random.choice(list(HYDROPHOBIC))
        
        elif aa_type == "polar":
            
            if random.random() < 0.5:
                return random.choice(list(HYDROPHOBIC))
            else:
                return aa
        
        else:
            
            return aa
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================================

# ============================================================================

def extract_contact_pairs_from_pdb(
    pdb_file: str,
    receptor_chain: str,
    binder_chain: str,
    contact_cutoff: float = 8.0,  
    use_cb: bool = True,  
    verbose: bool = True,
) -> List[Tuple[int, int]]:
    """extract contact pairs from pdb."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"extractcontactvs(mode1:autofromPDBextract)")
        print(f"{'='*60}")
        print(f"PDBfile: {pdb_file}")
        print(f"receptorchain: {receptor_chain}")
        print(f"chain: {binder_chain}")
        print(f"contactthreshold: {contact_cutoff} A (Angstrom)")
        print(f"useCBatom: {use_cb}")
    
    try:
        from Bio.PDB import PDBParser, MMCIFParser
    except ImportError:
        raise ImportError("BioPython is required. Install with: pip install biopython")
    
    
    if pdb_file.endswith('.cif') or pdb_file.endswith('.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure('structure', pdb_file)
    
    
    receptor_seq, receptor_ca_coords, receptor_res_map = parse_pdb_chain(
        pdb_file, receptor_chain, verbose=False
    )
    binder_seq, binder_ca_coords, binder_res_map = parse_pdb_chain(
        pdb_file, binder_chain, verbose=False
    )
    
    
    receptor_cb_coords = None
    binder_cb_coords = None
    
    if use_cb:
        try:
            receptor_cb_coords = _extract_cb_coords(structure, receptor_chain, len(receptor_seq))
            binder_cb_coords = _extract_cb_coords(structure, binder_chain, len(binder_seq))
        except Exception as e:
            if verbose:
                print(f"[WARNING] noextractCBcoordinates,useCAcoordinates: {e}")
            use_cb = False
    
    
    receptor_coords = receptor_cb_coords if (use_cb and receptor_cb_coords is not None) else receptor_ca_coords
    binder_coords = binder_cb_coords if (use_cb and binder_cb_coords is not None) else binder_ca_coords
    
    
    receptor_coords_np = receptor_coords.cpu().numpy() if isinstance(receptor_coords, torch.Tensor) else receptor_coords
    binder_coords_np = binder_coords.cpu().numpy() if isinstance(binder_coords, torch.Tensor) else binder_coords
    
    distances = np.sqrt(((receptor_coords_np[:, None, :] - binder_coords_np[None, :, :]) ** 2).sum(axis=2))
    # distances: [receptor_len, binder_len]
    
    
    
    
    
    

    binder_len = distances.shape[1]
    binder_idx_all = np.arange(binder_len, dtype=int)

    nearest_receptor = np.argmin(distances, axis=0)  # [binder_len]
    min_distance = distances[nearest_receptor, binder_idx_all]  # [binder_len]

    mask = min_distance < contact_cutoff
    if mask.any():
        cand_b = binder_idx_all[mask]
        cand_r = nearest_receptor[mask]
        cand_d = min_distance[mask]

        
        order = np.lexsort((cand_d, cand_r))  # sort by receptor then distance
        cand_r_sorted = cand_r[order]
        _, first_idx = np.unique(cand_r_sorted, return_index=True)
        chosen = order[first_idx]

        contact_pairs = list(zip(cand_r[chosen].tolist(), cand_b[chosen].tolist()))
    else:
        contact_pairs = []
    
    if verbose:
        print(f"\n[OK] extractto {len(contact_pairs)} contactvs")
        print(f"\n10contactvs(receptor_idx, binder_idx, distance):")
        for i, (r_idx, b_idx) in enumerate(contact_pairs[:10]):
            dist = distances[r_idx, b_idx]
            print(f"  {i+1}. ({r_idx}, {b_idx}) = {dist:.2f} A")
        if len(contact_pairs) > 10:
            print(f"... ( {len(contact_pairs)} vs)")
    
    return contact_pairs


def _extract_cb_coords(structure, chain_id: str, seq_length: int) -> Optional[torch.Tensor]:
    """ extract cb coords."""
    coords_list = []
    
    try:
        for model in structure:
            if chain_id not in model:
                return None
            
            chain = model[chain_id]
            residue_list = list(chain.get_residues())
            
            if len(residue_list) < seq_length:
                return None
            
            for res_idx, residue in enumerate(residue_list[:seq_length]):
                
                if 'CB' in residue:
                    atom = residue['CB']
                elif 'CA' in residue:
                    atom = residue['CA']
                else:
                    
                    atoms = list(residue.get_atoms())
                    if len(atoms) == 0:
                        return None
                    atom = atoms[0]
                
                coords_list.append(atom.coord)
            
            break  
        
        if len(coords_list) < seq_length:
            return None
        
        
        coords_array = np.array(coords_list, dtype=np.float32)
        return torch.tensor(coords_array, dtype=torch.float32)[:seq_length]
    
    except Exception as e:
        
        return None


# ============================================================================

# ============================================================================

def _generate_candidate_mappings(
    original_aa_r: str,
    original_aa_b: str,
    mode: str = "complementary",
) -> List[Tuple[Tuple[str, str], float]]:
    """ generate candidate mappings."""
    candidates = []
    receptor_type = classify_aa(original_aa_r)
    binder_type = classify_aa(original_aa_b)
    
    is_charged_pair = (receptor_type in ["positive", "negative"] and 
                      binder_type in ["positive", "negative"])
    is_hydrophobic_pair = (receptor_type == "hydrophobic" and 
                           binder_type == "hydrophobic")
    is_polar_hydrophobic = ((receptor_type == "polar" and binder_type == "hydrophobic") or
                            (receptor_type == "hydrophobic" and binder_type == "polar"))
    is_charged_polar = ((receptor_type in ["positive", "negative"] and binder_type == "polar") or
                        (receptor_type == "polar" and binder_type in ["positive", "negative"]))
    
    if mode == "complementary":
        if is_charged_pair:
            
            
            choices_b = [aa for aa in HYDROPHOBIC if aa != original_aa_b and aa not in {'C', 'P'}]
            if not choices_b:
                choices_b = [aa for aa in MUTABLE_AA if aa != original_aa_b and classify_aa(aa) == "hydrophobic"]
            if choices_b:
                aa_b = random.choice(choices_b)
                excluded_r = {original_aa_r, original_aa_b, aa_b}
                aa_r = get_complementary_aa(aa_b, mode="complementary")
                if aa_r in excluded_r or aa_r in {'C', 'P'}:
                    choices_r = [aa for aa in HYDROPHOBIC if aa not in excluded_r and aa not in {'C', 'P'} and classify_aa(aa) == "hydrophobic"]
                    if choices_r:
                        aa_r = random.choice(choices_r)
                    else:
                        choices_r = [aa for aa in MUTABLE_AA if aa not in excluded_r]
                        if choices_r:
                            aa_r = random.choice(choices_r)
                if aa_r != original_aa_r and aa_b != original_aa_b:
                    candidates.append(((aa_r, aa_b), 1.0))
                    
        elif is_hydrophobic_pair:
            
            
            choices_b_charged = [aa for aa in (POSITIVE_CHARGED | NEGATIVE_CHARGED) if aa != original_aa_b]
            if choices_b_charged:
                aa_b1 = random.choice(choices_b_charged)
                excluded_r1 = {original_aa_r, original_aa_b, aa_b1}
                aa_r1 = get_complementary_aa(aa_b1, mode="complementary")
                if aa_r1 in excluded_r1 or aa_r1 in {'C', 'P'}:
                    if classify_aa(aa_b1) == "positive":
                        choices_r1 = [aa for aa in NEGATIVE_CHARGED if aa not in excluded_r1]
                    else:
                        choices_r1 = [aa for aa in POSITIVE_CHARGED if aa not in excluded_r1]
                    if choices_r1:
                        aa_r1 = random.choice(choices_r1)
                    else:
                        choices_r1 = [aa for aa in MUTABLE_AA if aa not in excluded_r1]
                        if choices_r1:
                            aa_r1 = random.choice(choices_r1)
                if aa_r1 != original_aa_r and aa_b1 != original_aa_b:
                    candidates.append(((aa_r1, aa_b1), 0.8))
            
            
            choices_b_hydro = [aa for aa in HYDROPHOBIC if aa != original_aa_b and aa not in {'C', 'P'}]
            if choices_b_hydro:
                aa_b2 = random.choice(choices_b_hydro)
                excluded_r2 = {original_aa_r, original_aa_b, aa_b2}
                aa_r2 = get_complementary_aa(aa_b2, mode="complementary")
                if aa_r2 in excluded_r2 or aa_r2 in {'C', 'P'}:
                    choices_r2 = [aa for aa in HYDROPHOBIC if aa not in excluded_r2 and aa not in {'C', 'P'} and classify_aa(aa) == "hydrophobic"]
                    if choices_r2:
                        aa_r2 = random.choice(choices_r2)
                    else:
                        choices_r2 = [aa for aa in MUTABLE_AA if aa not in excluded_r2]
                        if choices_r2:
                            aa_r2 = random.choice(choices_r2)
                if aa_r2 != original_aa_r and aa_b2 != original_aa_b:
                    candidates.append(((aa_r2, aa_b2), 0.2))
                    
        elif is_polar_hydrophobic:
            
            
            choices_b_charged = [aa for aa in (POSITIVE_CHARGED | NEGATIVE_CHARGED) if aa != original_aa_b]
            if choices_b_charged:
                aa_b1 = random.choice(choices_b_charged)
                excluded_r1 = {original_aa_r, original_aa_b, aa_b1}
                aa_r1 = get_complementary_aa(aa_b1, mode="complementary")
                if aa_r1 in excluded_r1 or aa_r1 in {'C', 'P'}:
                    if classify_aa(aa_b1) == "positive":
                        choices_r1 = [aa for aa in NEGATIVE_CHARGED if aa not in excluded_r1]
                    else:
                        choices_r1 = [aa for aa in POSITIVE_CHARGED if aa not in excluded_r1]
                    if choices_r1:
                        aa_r1 = random.choice(choices_r1)
                    else:
                        choices_r1 = [aa for aa in MUTABLE_AA if aa not in excluded_r1]
                        if choices_r1:
                            aa_r1 = random.choice(choices_r1)
                if aa_r1 != original_aa_r and aa_b1 != original_aa_b:
                    candidates.append(((aa_r1, aa_b1), 0.8))
            
            
            choices_b_other = [aa for aa in MUTABLE_AA if aa != original_aa_b]
            if choices_b_other:
                aa_b2 = random.choice(choices_b_other)
                excluded_r2 = {original_aa_r, original_aa_b, aa_b2}
                aa_r2 = get_complementary_aa(aa_b2, mode="complementary")
                if aa_r2 in excluded_r2 or aa_r2 in {'C', 'P'}:
                    choices_r2 = [aa for aa in MUTABLE_AA if aa not in excluded_r2]
                    if choices_r2:
                        aa_r2 = random.choice(choices_r2)
                if aa_r2 != original_aa_r and aa_b2 != original_aa_b:
                    candidates.append(((aa_r2, aa_b2), 0.2))
                    
        elif is_charged_polar:
            
            
            if receptor_type in ["positive", "negative"]:
                if receptor_type == "positive":
                    receptor_choices = [aa for aa in NEGATIVE_CHARGED if aa != original_aa_r]
                else:
                    receptor_choices = [aa for aa in POSITIVE_CHARGED if aa != original_aa_r]
                if receptor_choices:
                    aa_r1 = random.choice(receptor_choices)
                    excluded_b1 = {original_aa_r, original_aa_b, aa_r1}
                    aa_b1 = get_complementary_aa(aa_r1, mode="complementary")
                    if aa_b1 in excluded_b1 or aa_b1 in {'C', 'P'}:
                        if classify_aa(aa_r1) == "positive":
                            choices_b1 = [aa for aa in NEGATIVE_CHARGED if aa not in excluded_b1]
                        else:
                            choices_b1 = [aa for aa in POSITIVE_CHARGED if aa not in excluded_b1]
                        if choices_b1:
                            aa_b1 = random.choice(choices_b1)
                        else:
                            choices_b1 = [aa for aa in MUTABLE_AA if aa not in excluded_b1]
                            if choices_b1:
                                aa_b1 = random.choice(choices_b1)
                    if aa_r1 != original_aa_r and aa_b1 != original_aa_b:
                        candidates.append(((aa_r1, aa_b1), 0.7))
            else:
                
                if binder_type == "positive":
                    binder_choices = [aa for aa in NEGATIVE_CHARGED if aa != original_aa_b]
                else:
                    binder_choices = [aa for aa in POSITIVE_CHARGED if aa != original_aa_b]
                if binder_choices:
                    aa_b1 = random.choice(binder_choices)
                    excluded_r1 = {original_aa_r, original_aa_b, aa_b1}
                    aa_r1 = get_complementary_aa(aa_b1, mode="complementary")
                    if aa_r1 in excluded_r1 or aa_r1 in {'C', 'P'}:
                        if classify_aa(aa_b1) == "positive":
                            choices_r1 = [aa for aa in NEGATIVE_CHARGED if aa not in excluded_r1]
                        else:
                            choices_r1 = [aa for aa in POSITIVE_CHARGED if aa not in excluded_r1]
                        if choices_r1:
                            aa_r1 = random.choice(choices_r1)
                        else:
                            choices_r1 = [aa for aa in MUTABLE_AA if aa not in excluded_r1]
                            if choices_r1:
                                aa_r1 = random.choice(choices_r1)
                    if aa_r1 != original_aa_r and aa_b1 != original_aa_b:
                        candidates.append(((aa_r1, aa_b1), 0.7))
            
            
            choices_r_hydro = [aa for aa in HYDROPHOBIC if aa != original_aa_r and aa not in {'C', 'P'}]
            if choices_r_hydro:
                aa_r2 = random.choice(choices_r_hydro)
                excluded_b2 = {original_aa_r, original_aa_b, aa_r2}
                aa_b2 = get_complementary_aa(aa_r2, mode="complementary")
                if aa_b2 in excluded_b2 or aa_b2 in {'C', 'P'}:
                    choices_b2 = [aa for aa in HYDROPHOBIC if aa not in excluded_b2 and aa not in {'C', 'P'}]
                    if choices_b2:
                        aa_b2 = random.choice(choices_b2)
                    else:
                        choices_b2 = [aa for aa in MUTABLE_AA if aa not in excluded_b2]
                        if choices_b2:
                            aa_b2 = random.choice(choices_b2)
                if aa_r2 != original_aa_r and aa_b2 != original_aa_b:
                    candidates.append(((aa_r2, aa_b2), 0.3))
        else:
            
            choices_b = [aa for aa in MUTABLE_AA if aa != original_aa_b]
            if choices_b:
                aa_b = random.choice(choices_b)
                excluded_r = {original_aa_r, original_aa_b, aa_b}
                aa_r = get_complementary_aa(aa_b, mode="complementary")
                if aa_r in excluded_r or aa_r in {'C', 'P'}:
                    choices_r = [aa for aa in MUTABLE_AA if aa not in excluded_r]
                    if choices_r:
                        aa_r = random.choice(choices_r)
                if aa_r != original_aa_r or aa_b != original_aa_b:
                    candidates.append(((aa_r, aa_b), 1.0))
    
    
    if candidates:
        total_prob = sum(prob for _, prob in candidates)
        if total_prob > 0:
            candidates = [((aa_r, aa_b), prob / total_prob) for (aa_r, aa_b), prob in candidates]
        else:
            
            candidates = [((aa_r, aa_b), 1.0 / len(candidates)) for (aa_r, aa_b), _ in candidates]
    else:
        
        aa_b = random.choice([aa for aa in MUTABLE_AA if aa != original_aa_b] or ["A"])
        aa_r = get_complementary_aa(aa_b, mode="complementary")
        if aa_r == original_aa_r or aa_r in {'C', 'P'}:
            aa_r = random.choice([aa for aa in MUTABLE_AA if aa != original_aa_r] or ["A"])
        candidates.append(((aa_r, aa_b), 1.0))
    
    return candidates


def build_contact_pair_mapping(
    contact_pairs: List[Tuple[int, int]],
    receptor_seq: str,
    binder_seq: str,
    contact_distances: Optional[Dict[Tuple[int, int], float]] = None,
    mode: str = "complementary",  # "same", "complementary", "distance_based"
    verbose: bool = True,
) -> Dict[Tuple[int, int], List[Tuple[Tuple[str, str], float]]]:
    """build contact pair mapping."""
    mapping = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"contactvs(mode: {mode})")
        print(f"{'='*60}")
    
    for (receptor_idx, binder_idx) in contact_pairs:
        
        original_aa_r = receptor_seq[receptor_idx].upper()
        original_aa_b = binder_seq[binder_idx].upper()
        
        
        if mode == "same":
            
            
            excluded = {original_aa_r, original_aa_b, 'C', 'P'}
            available_aas = [aa for aa in MUTABLE_AA if aa not in excluded]
            if not available_aas:
                available_aas = list(MUTABLE_AA)  
            
            total = len(available_aas)
            candidates = [((aa, aa), 1.0 / total) for aa in available_aas]
        
        elif mode == "complementary":
            
            candidates = _generate_candidate_mappings(original_aa_r, original_aa_b, mode="complementary")
        
        elif mode == "distance_based":
            
            if contact_distances and (receptor_idx, binder_idx) in contact_distances:
                distance = contact_distances[(receptor_idx, binder_idx)]
                if distance < 4.0:  
                    candidates = _generate_candidate_mappings(original_aa_r, original_aa_b, mode="complementary")
                else:  
                    excluded = {original_aa_r, original_aa_b, 'C', 'P'}
                    available_aas = [aa for aa in MUTABLE_AA if aa not in excluded]
                    if not available_aas:
                        available_aas = list(MUTABLE_AA)
                    total = len(available_aas)
                    candidates = [((aa, aa), 1.0 / total) for aa in available_aas]
            else:
                
                candidates = _generate_candidate_mappings(original_aa_r, original_aa_b, mode="complementary")
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        
        mapping[(receptor_idx, binder_idx)] = candidates
        
    
    total_candidates = sum(len(candidates) for candidates in mapping.values())
    both_changed_candidates = 0
    for (r_idx, b_idx), candidates in mapping.items():
        for (aa_r, aa_b), _ in candidates:
            if aa_r != receptor_seq[r_idx] and aa_b != binder_seq[b_idx]:
                both_changed_candidates += 1
    
    if verbose:
        print(f"\n[OK] {len(mapping)} contactvs")
        print(f" -: {total_candidates}")
        print(f" -: {both_changed_candidates} ")
    
    return mapping


# ============================================================================

# ============================================================================

def generate_fake_paired_msa_csv(
    receptor_seq: str,
    binder_seq: str,
    contact_pairs: List[Tuple[int, int]],
    contact_pair_mapping: Dict[Tuple[int, int], List[Tuple[Tuple[str, str], float]]],
    num_seqs: int = 100,
    correlation_strength: float = 0.8,
    non_contact_mutation_rate: float = 0.02,  
    use_blosum62: bool = True,                 
    taxonomy_id: int = 999999,
    output_dir: Optional[Path] = None,
    receptor_chain_id: str = "B",
    binder_chain_id: str = "A",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generate fake paired msa csv."""
    
    if num_seqs < MIN_PAIRED_MSA_NUM_SEQS:
        num_seqs = MIN_PAIRED_MSA_NUM_SEQS

    L_r = len(receptor_seq)
    L_b = len(binder_seq)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"generate Paired MSA(Boltz2:CSVfile)")
        print(f"{'='*60}")
        print(f"Receptorlength: {L_r}")
        print(f"Binderlength: {L_b}")
        print(f"contactvs: {len(contact_pairs)}")
        print(f"sequence: {num_seqs}")
        print(f": {correlation_strength}")
        print(f"contactpositionmutation: {non_contact_mutation_rate*100:.1f}%")
        print(f"useBLOSUM62mutation: {'' if use_blosum62 else ''}")
        print(f"Taxonomy ID: {taxonomy_id}")
    
    
    receptor_sequences = []
    binder_sequences = []
    keys = []
    
    
    receptor_sequences.append(receptor_seq)
    binder_sequences.append(binder_seq)
    keys.append(taxonomy_id)
    
    if verbose:
        print(f"\n[OK] firstsequence():")
        print(f"   Receptor: {receptor_seq[:50]}...")
        print(f"   Binder: {binder_seq[:50]}...")
    
    
    receptor_contact_indices = {idx for idx, _ in contact_pairs}
    binder_contact_indices = {idx for _, idx in contact_pairs}
    
    for k in range(1, num_seqs):
        seq_r = list(receptor_seq)
        seq_b = list(binder_seq)
        
        
        for (idx_r, idx_b) in contact_pairs:
            if random.random() < correlation_strength:
                
                if (idx_r, idx_b) in contact_pair_mapping:
                    candidates = contact_pair_mapping[(idx_r, idx_b)]
                    
                    if candidates and len(candidates) > 0:
                        
                        candidate_pairs = [pair for pair, _ in candidates]
                        weights = [prob for _, prob in candidates]
                        
                        total_weight = sum(weights)
                        if total_weight > 0:
                            normalized_weights = [w / total_weight for w in weights]
                            
                            selected_idx = np.random.choice(len(candidate_pairs), p=normalized_weights)
                            aa_r, aa_b = candidate_pairs[selected_idx]
                        else:
                            
                            aa_r, aa_b = random.choice(candidate_pairs)
                        
                        seq_r[idx_r] = aa_r
                        seq_b[idx_b] = aa_b
                    else:
                        
                        aa = random.choice(MUTABLE_AA)
                        seq_r[idx_r] = aa
                        seq_b[idx_b] = aa
                else:
                    
                    aa = random.choice(MUTABLE_AA)
                    seq_r[idx_r] = aa
                    seq_b[idx_b] = aa
            else:
                
                if use_blosum62:
                    
                    seq_r[idx_r] = sample_conservative_mutation(seq_r[idx_r])
                    seq_b[idx_b] = sample_conservative_mutation(seq_b[idx_b])
                else:
                    
                    seq_r[idx_r] = random.choice(MUTABLE_AA)
                    seq_b[idx_b] = random.choice(MUTABLE_AA)
        
        
        for i in range(L_r):
            if i not in receptor_contact_indices:
                if random.random() < non_contact_mutation_rate:
                    if use_blosum62:
                        
                        seq_r[i] = sample_conservative_mutation(seq_r[i])
                    else:
                        
                        seq_r[i] = random.choice(MUTABLE_AA)
        
        for i in range(L_b):
            if i not in binder_contact_indices:
                if random.random() < non_contact_mutation_rate:
                    if use_blosum62:
                        
                        seq_b[i] = sample_conservative_mutation(seq_b[i])
                    else:
                        
                        seq_b[i] = random.choice(MUTABLE_AA)
        
        
        receptor_sequences.append(''.join(seq_r))
        binder_sequences.append(''.join(seq_b))
        keys.append(taxonomy_id)
    
    
    receptor_df = pd.DataFrame({
        'key': keys,
        'sequence': receptor_sequences,
    })
    binder_df = pd.DataFrame({
        'key': keys,
        'sequence': binder_sequences,
    })
    
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        receptor_path = output_dir / f"receptor_{receptor_chain_id}.csv"
        binder_path = output_dir / f"binder_{binder_chain_id}.csv"
        receptor_df.to_csv(receptor_path, index=False)
        binder_df.to_csv(binder_path, index=False)
        if verbose:
            print(f"\n[OK] MSAalreadysaveto:")
            print(f"   Receptor: {receptor_path}")
            print(f"   Binder: {binder_path}")
    
    if verbose:
        print(f"\n[OK] okgenerate {len(receptor_df)} vssequence")
        print(f" - Receptorfirstsequence: {receptor_df.iloc[0]['sequence'][:50]}...")
        print(f" - Binderfirstsequence: {binder_df.iloc[0]['sequence'][:50]}...")
    
    return receptor_df, binder_df


# ============================================================================

# ============================================================================

def generate_fake_msa_mode1(
    pdb_file: str,
    receptor_chain: str,
    binder_chain: str,
    contact_cutoff: float = 8.0,
    use_cb: bool = True,
    mapping_mode: str = "complementary",
    num_seqs: int = 100,
    correlation_strength: float = 0.8,
    non_contact_mutation_rate: float = 0.02,  
    use_blosum62: bool = True,                 
    taxonomy_id: int = 999999,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generate fake msa mode1."""
    
    if num_seqs < MIN_PAIRED_MSA_NUM_SEQS:
        num_seqs = MIN_PAIRED_MSA_NUM_SEQS

    
    receptor_seq, _, _ = parse_pdb_chain(pdb_file, receptor_chain, verbose=verbose)
    binder_seq, _, _ = parse_pdb_chain(pdb_file, binder_chain, verbose=verbose)
    
    
    contact_pairs = extract_contact_pairs_from_pdb(
        pdb_file=pdb_file,
        receptor_chain=receptor_chain,
        binder_chain=binder_chain,
        contact_cutoff=contact_cutoff,
        use_cb=use_cb,
        verbose=verbose,
    )
    
    if len(contact_pairs) == 0:
        raise ValueError(f"nottocontactvs！checkcontactthreshold(current: {contact_cutoff} A)")
    
    
    contact_distances = None
    if mapping_mode == "distance_based":
        
        receptor_seq_check, receptor_ca_coords, _ = parse_pdb_chain(pdb_file, receptor_chain, verbose=False)
        binder_seq_check, binder_ca_coords, _ = parse_pdb_chain(pdb_file, binder_chain, verbose=False)
        
        receptor_coords = receptor_ca_coords.cpu().numpy() if isinstance(receptor_ca_coords, torch.Tensor) else receptor_ca_coords
        binder_coords = binder_ca_coords.cpu().numpy() if isinstance(binder_ca_coords, torch.Tensor) else binder_ca_coords
        
        
        distances = np.sqrt(((receptor_coords[:, None, :] - binder_coords[None, :, :]) ** 2).sum(axis=2))
        
        
        contact_distances = {}
        for (r_idx, b_idx) in contact_pairs:
            if r_idx < distances.shape[0] and b_idx < distances.shape[1]:
                contact_distances[(r_idx, b_idx)] = float(distances[r_idx, b_idx])
    
    
    contact_pair_mapping = build_contact_pair_mapping(
        contact_pairs=contact_pairs,
        receptor_seq=receptor_seq,
        binder_seq=binder_seq,
        contact_distances=contact_distances,
        mode=mapping_mode,
        verbose=verbose,
    )
    
    
    receptor_df, binder_df = generate_fake_paired_msa_csv(
        receptor_seq=receptor_seq,
        binder_seq=binder_seq,
        contact_pairs=contact_pairs,
        contact_pair_mapping=contact_pair_mapping,
        num_seqs=num_seqs,
        correlation_strength=correlation_strength,
        non_contact_mutation_rate=non_contact_mutation_rate,
        use_blosum62=use_blosum62,
        taxonomy_id=taxonomy_id,
        output_dir=output_dir,
        receptor_chain_id=receptor_chain,
        binder_chain_id=binder_chain,
        verbose=verbose,
    )
    
    return receptor_df, binder_df


def generate_fake_msa_mode2(
    receptor_seq: str,
    binder_seq: str,
    contact_pairs: List[Tuple[int, int]],  
    mapping_mode: str = "complementary",
    contact_distances: Optional[Dict[Tuple[int, int], float]] = None,
    num_seqs: int = 100,
    correlation_strength: float = 0.8,
    non_contact_mutation_rate: float = 0.02,  
    use_blosum62: bool = True,                 
    taxonomy_id: int = 999999,
    output_dir: Optional[Path] = None,
    receptor_chain_id: str = "B",
    binder_chain_id: str = "A",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generate fake msa mode2."""
    
    if num_seqs < MIN_PAIRED_MSA_NUM_SEQS:
        num_seqs = MIN_PAIRED_MSA_NUM_SEQS

    
    if len(receptor_seq) == 0 or len(binder_seq) == 0:
        raise ValueError('sequencenotasempty！')
    
    if len(contact_pairs) == 0:
        raise ValueError('contactvslistnotasempty！')
    
    
    for (r_idx, b_idx) in contact_pairs:
        if r_idx < 0 or r_idx >= len(receptor_seq):
            raise ValueError(f"Receptorindex {r_idx} [0, {len(receptor_seq)})")
        if b_idx < 0 or b_idx >= len(binder_seq):
            raise ValueError(f"Binderindex {b_idx} [0, {len(binder_seq)})")
    
    
    contact_pair_mapping = build_contact_pair_mapping(
        contact_pairs=contact_pairs,
        receptor_seq=receptor_seq,
        binder_seq=binder_seq,
        contact_distances=contact_distances,
        mode=mapping_mode,
        verbose=verbose,
    )
    
    
    receptor_df, binder_df = generate_fake_paired_msa_csv(
        receptor_seq=receptor_seq,
        binder_seq=binder_seq,
        contact_pairs=contact_pairs,
        contact_pair_mapping=contact_pair_mapping,
        num_seqs=num_seqs,
        correlation_strength=correlation_strength,
        non_contact_mutation_rate=non_contact_mutation_rate,
        use_blosum62=use_blosum62,
        taxonomy_id=taxonomy_id,
        output_dir=output_dir,
        receptor_chain_id=receptor_chain_id,
        binder_chain_id=binder_chain_id,
        verbose=verbose,
    )
    
    return receptor_df, binder_df


# ============================================================================

# ============================================================================

def generate_fake_msa_conservative(
    pdb_file: str,
    receptor_chain: str,
    binder_chain: str,
    contact_cutoff: float = 8.0,
    use_cb: bool = True,
    mapping_mode: str = "complementary",
    num_seqs: int = 500,
    correlation_strength: float = 0.8,
    non_contact_mutation_rate: float = 0.02,  
    use_blosum62: bool = True,                
    taxonomy_id: int = 999999,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generate fake msa conservative."""
    
    if num_seqs < MIN_PAIRED_MSA_NUM_SEQS:
        num_seqs = MIN_PAIRED_MSA_NUM_SEQS

    return generate_fake_msa_mode1(
        pdb_file=pdb_file,
        receptor_chain=receptor_chain,
        binder_chain=binder_chain,
        contact_cutoff=contact_cutoff,
        use_cb=use_cb,
        mapping_mode=mapping_mode,
        num_seqs=num_seqs,
        correlation_strength=correlation_strength,
        non_contact_mutation_rate=non_contact_mutation_rate,
        use_blosum62=use_blosum62,
        taxonomy_id=taxonomy_id,
        output_dir=output_dir,
        verbose=verbose,
    )


# ============================================================================

# ============================================================================

class MSAGeneratorData:
    """MSAGeneratorData."""
    def __init__(
        self,
        receptor_seq: str,
        contact_pairs: List[Tuple[int, int]],
        contact_pair_mapping: Dict[Tuple[int, int], List[Tuple[Tuple[str, str], float]]],
        receptor_chain_id: str = "B",
        binder_chain_id: str = "A",
        original_binder_seq: Optional[str] = None,  
    ):
        """  init  ."""
        self.receptor_seq = receptor_seq
        self.contact_pairs = contact_pairs
        self.contact_pair_mapping = contact_pair_mapping
        self.receptor_chain_id = receptor_chain_id
        self.binder_chain_id = binder_chain_id
        self.original_binder_seq = original_binder_seq  


def prepare_msa_generator_data(
    pdb_file: str,
    receptor_chain: str,
    binder_chain: str,
    contact_cutoff: float = 8.0,
    use_cb: bool = True,
    mapping_mode: str = "complementary",
    verbose: bool = True,
) -> MSAGeneratorData:
    """prepare msa generator data."""
    
    receptor_seq, _, _ = parse_pdb_chain(pdb_file, receptor_chain, verbose=verbose)
    
    
    original_binder_seq, _, _ = parse_pdb_chain(pdb_file, binder_chain, verbose=verbose)
    
    
    contact_pairs = extract_contact_pairs_from_pdb(
        pdb_file=pdb_file,
        receptor_chain=receptor_chain,
        binder_chain=binder_chain,
        contact_cutoff=contact_cutoff,
        use_cb=use_cb,
        verbose=verbose,
    )
    
    if len(contact_pairs) == 0:
        raise ValueError(f"nottocontactvs！checkcontactthreshold(current: {contact_cutoff} A)")
    
    
    contact_distances = None
    if mapping_mode == "distance_based":
        receptor_seq_check, receptor_ca_coords, _ = parse_pdb_chain(pdb_file, receptor_chain, verbose=False)
        binder_seq_check, binder_ca_coords, _ = parse_pdb_chain(pdb_file, binder_chain, verbose=False)
        
        receptor_coords = receptor_ca_coords.cpu().numpy() if isinstance(receptor_ca_coords, torch.Tensor) else receptor_ca_coords
        binder_coords = binder_ca_coords.cpu().numpy() if isinstance(binder_ca_coords, torch.Tensor) else binder_ca_coords
        
        distances = np.sqrt(((receptor_coords[:, None, :] - binder_coords[None, :, :]) ** 2).sum(axis=2))
        
        contact_distances = {}
        for (r_idx, b_idx) in contact_pairs:
            if r_idx < distances.shape[0] and b_idx < distances.shape[1]:
                contact_distances[(r_idx, b_idx)] = float(distances[r_idx, b_idx])
    
    
    contact_pair_mapping = build_contact_pair_mapping(
        contact_pairs=contact_pairs,
        receptor_seq=receptor_seq,
        binder_seq=original_binder_seq,  
        contact_distances=contact_distances,
        mode=mapping_mode,
        verbose=verbose,
    )
    
    if verbose:
        print(f"\n[OK] MSAgenerate")
        print(f" - Receptorsequencelength: {len(receptor_seq)}")
        print(f" - contactdistancethreshold: {contact_cutoff} A")
        print(f" - contactvs: {len(contact_pairs)}")
        print(f" - mode: {mapping_mode}")
    
    return MSAGeneratorData(
        receptor_seq=receptor_seq,
        contact_pairs=contact_pairs,
        contact_pair_mapping=contact_pair_mapping,
        receptor_chain_id=receptor_chain,
        binder_chain_id=binder_chain,
        original_binder_seq=original_binder_seq,  
    )


def generate_dynamic_msa(
    msa_data: MSAGeneratorData,
    binder_seq: str,
    num_seqs: int = 500,
    correlation_strength: float = 0.8,
    non_contact_mutation_rate: float = 0.02,
    use_blosum62: bool = True,
    taxonomy_id: int = 999999,
    output_dir: Optional[Path] = None,
    save_temp_files: bool = False,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generate dynamic msa."""
    
    if num_seqs < MIN_PAIRED_MSA_NUM_SEQS:
        num_seqs = MIN_PAIRED_MSA_NUM_SEQS

    
    
    if msa_data.original_binder_seq is not None:
        expected_binder_length = len(msa_data.original_binder_seq)
    elif msa_data.contact_pairs:
        max_binder_idx = max(b_idx for _, b_idx in msa_data.contact_pairs)
        expected_binder_length = max_binder_idx + 1
    else:
        
        expected_binder_length = len(binder_seq)
    
    if len(binder_seq) != expected_binder_length:
        raise ValueError(
            f"Bindersequencelength ({len(binder_seq)}) andlength ({expected_binder_length}) notmatch！"
            f"contactvsindexPDBbindersequence,needsequencelength."
            f"bindersequencelength: {len(msa_data.original_binder_seq) if msa_data.original_binder_seq else 'not'}"
        )
    
    
    temp_output_dir = None
    if save_temp_files:
        if output_dir is None:
            import tempfile
            temp_output_dir = Path(tempfile.mkdtemp(prefix="msa_temp_"))
        else:
            temp_output_dir = Path(output_dir)
            temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    
    receptor_df, binder_df = generate_fake_paired_msa_csv(
        receptor_seq=msa_data.receptor_seq,  
        binder_seq=binder_seq,  
        contact_pairs=msa_data.contact_pairs,  
        contact_pair_mapping=msa_data.contact_pair_mapping,  
        num_seqs=num_seqs,
        correlation_strength=correlation_strength,
        non_contact_mutation_rate=non_contact_mutation_rate,
        use_blosum62=use_blosum62,
        taxonomy_id=taxonomy_id,
        output_dir=temp_output_dir,
        receptor_chain_id=msa_data.receptor_chain_id,
        binder_chain_id=msa_data.binder_chain_id,
        verbose=verbose,
    )
    
    return receptor_df, binder_df


# ============================================================================

# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print('1:mode1(autofromPDBextractcontactvs)')
    print("="*60)
    
    # fake_msa_df = generate_fake_msa_mode1(
    #     pdb_file="path/to/complex.pdb",
    #     receptor_chain="B",
    #     binder_chain="A",
    #     contact_cutoff=8.0,
    #     use_cb=True,
    #     mapping_mode="complementary",
    #     num_seqs=100,
    #     correlation_strength=0.8,
    #     output_path=Path("fake_msa_mode1.csv"),
    #     verbose=True,
    # )
    
    
    print("\n" + "="*60)
    print('2:mode2(contactvs)')
    print("="*60)
    
    receptor_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    binder_seq = "ACDEFGHIKLMNPQRSTVWY"
    contact_pairs = [(10, 5), (15, 8), (20, 12)]  
    
    receptor_df, binder_df = generate_fake_msa_mode2(
        receptor_seq=receptor_seq,
        binder_seq=binder_seq,
        contact_pairs=contact_pairs,
        mapping_mode="complementary",
        num_seqs=50,
        correlation_strength=0.8,
        output_dir=Path("fake_msa_mode2_output"),
        verbose=True,
    )
    
    print('\n[OK] ！')

