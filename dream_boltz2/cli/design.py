#!/usr/bin/env python3
"""Dreaming: optimize single and pair representations and sample backbones."""

import sys
from pathlib import Path

# ============================================================================

# ============================================================================
def _setup_python_path():
    """Add the repository root, and BOLTZ_SRC if set, to sys.path."""
    import os
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    boltz_src = os.environ.get("BOLTZ_SRC")
    if boltz_src and boltz_src not in sys.path:
        sys.path.insert(0, boltz_src)
    return root

_DREAM_BOLTZ2_ROOT = _setup_python_path()
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict, List

from dream_boltz2.model.diffusion import DiffusionWrapper
from dream_boltz2.model.pairformer import PairformerWrapper  
from dream_boltz2.features.complex import FeaturePreparatorComplex
from dream_boltz2.io.pdb import parse_pdb_complex
from boltz.model.models.boltz2 import Boltz2


try:
    from dream_boltz2.io.pdb import write_coords_to_pdb
except ImportError:
    
    write_coords_to_pdb = None
    print('[WARN] write_coords_to_pdbnot,PDBoutput')


def load_boltz_model(checkpoint_path: str, device: str = 'cuda', use_physical_guidance: bool = False) -> Boltz2:
    """load boltz model."""
    from boltz.main import (
        Boltz2DiffusionParams,
        PairformerArgsV2,
        MSAModuleArgs,
        BoltzSteeringParams
    )
    from dataclasses import asdict
    
    print(f"Loading Boltz2 model from: {checkpoint_path}")
    
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    pairformer_args.activation_checkpointing = False
    
    msa_args = MSAModuleArgs(
        subsample_msa=False,
        num_subsampled_msa=1024,
        use_paired_feature=True
    )
    
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = use_physical_guidance  
    
    if use_physical_guidance:
        print(' already(ConnectionsPotentialconstraintchain)')
    else:
        print(' already(foroptimizationstepgradient,generateanchor)')
    
    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 15,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    
    boltz_model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        predict_args=predict_args,
        map_location="cpu" if device == "cpu" else "cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=True,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    
    boltz_model = boltz_model.to(device)
    boltz_model.eval()
    
    
    for param in boltz_model.parameters():
        param.requires_grad = False
    
    print(' Model loaded and frozen')
    return boltz_model


# ============================================================================
# ============================================================================

# ============================================================================

def get_nanobody_template(template_name: str = 'nanobody_default') -> tuple[str, list]:
    """get nanobody template."""
    from dream_boltz2.sequence.cdr import get_nanobody_template as _get_nanobody_template
    return _get_nanobody_template(template_name)


def generate_nanobody_sequence(template_name: str = 'random') -> tuple[str, list, str]:
    """generate nanobody sequence."""
    from dream_boltz2.sequence.cdr import generate_nanobody_sequence as _generate
    return _generate(template_name)


def get_fab_template(template_name: str = 'fab_default') -> tuple[str, list, dict]:
    """get fab template."""
    if template_name == 'fab_default':
        
        
        
        
        h_chain = "EVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXWVRQAPGKGLEWVAXXXXXXXXXXXXXXXXXRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRXXXXXXXXXXXWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKS"
        l_chain = "DIQMTQSPSSLSASVGDRVTITCXXXXXXXXXXXWYQQKPGKAPKLLIYXXXXXXXGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCXXXXXXXXXFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGES"
        
        h_chain_length = len(h_chain)
        l_chain_length = len(l_chain)
        
        
        sequence = h_chain + l_chain
        total_length = len(sequence)
        
        
        def find_cdr_regions(seq: str, offset: int = 0) -> list:
            """find cdr regions."""
            cdr_regions = []
            start = None
            for i, aa in enumerate(seq):
                if aa == 'X':
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        cdr_regions.append((start + offset, i + offset))
                        start = None
            
            if start is not None:
                cdr_regions.append((start + offset, len(seq) + offset))
            return cdr_regions
        
        
        h_cdr_regions = find_cdr_regions(h_chain, offset=0)
        
        
        l_cdr_regions = find_cdr_regions(l_chain, offset=h_chain_length)
        
        
        cdr_regions = h_cdr_regions + l_cdr_regions
        
        
        chain_info = {
            'H_chain_length': h_chain_length,
            'L_chain_length': l_chain_length,
            'H_chain_slice': slice(0, h_chain_length),
            'L_chain_slice': slice(h_chain_length, total_length),
        }
        
        print(f"\n[Fab Template] {template_name}:")
        print(f" Hchainlength: {h_chain_length} residue")
        print(f" Lchainlength: {l_chain_length} residue")
        print(f" length: {total_length} residue")
        print(f" HchainCDR: {len(h_cdr_regions)}")
        for i, (start, end) in enumerate(h_cdr_regions, 1):
            print(f"    H-CDR{i}: [{start}, {end}) ({end - start} residue)")
        print(f" LchainCDR: {len(l_cdr_regions)}")
        for i, (start, end) in enumerate(l_cdr_regions, 1):
            print(f"    L-CDR{i}: [{start}, {end}) ({end - start} residue)")
        
        return sequence, cdr_regions, chain_info
    else:
        raise ValueError(f"Unknown Fab template: {template_name}")


def parse_fixed_residue_contacts(
    fixed_residue_contacts_str: str,
    receptor_seq: str,
    cdr_regions: list,
    binder_length: int,
    exclude_binder_positions: set = None,
    exclude_receptor_positions: set = None,
    verbose: bool = True,
) -> tuple:
    """parse fixed residue contacts."""
    import random
    
    fixed_residue_contacts = {}
    fixed_binder_positions = set() if exclude_binder_positions is None else set(exclude_binder_positions)
    fixed_receptor_positions = set() if exclude_receptor_positions is None else set(exclude_receptor_positions)
    fixed_binder_aa_map = {}
    fixed_contact_pairs = []
    seen_binder_positions = set(fixed_binder_positions)
    seen_receptor_positions = set(fixed_receptor_positions)
    
    
    cdr_name_to_index = {
        'CDR1': 0,
        'CDR2': 1,
        'CDR3': 2,
        'H_CDR1': 0, 'H-CDR1': 0,
        'H_CDR2': 1, 'H-CDR2': 1,
        'H_CDR3': 2, 'H-CDR3': 2,
        'L_CDR1': 3, 'L-CDR1': 3,
        'L_CDR2': 4, 'L-CDR2': 4,
        'L_CDR3': 5, 'L-CDR3': 5,
    }
    
    for pair_str in fixed_residue_contacts_str.split(','):
        pair_str = pair_str.strip()
        if '-' not in pair_str:
            if verbose:
                print(f" warning:nofixedresiduevs: {pair_str}")
            continue
        
        parts = pair_str.split('-')
        if len(parts) != 2:
            if verbose:
                print(f" warning:nofixedresiduevs: {pair_str}")
            continue
        
        binder_part = parts[0].strip()
        receptor_part = parts[1].strip()
        
        
        binder_idx = None
        binder_aa = None
        
        
        if binder_part[0].isdigit():
            try:
                binder_pos_1based = int(binder_part[:-1])
                binder_aa = binder_part[-1].upper()
                binder_idx = binder_pos_1based - 1  
            except ValueError:
                if verbose:
                    print(f" warning:noparsebinderposition: {binder_part}")
                continue
        
        elif binder_part.upper().startswith('CDR'):
            
            
            cdr_name_match = None
            for cdr_name, cdr_idx in cdr_name_to_index.items():
                if binder_part.upper().startswith(cdr_name.upper()):
                    cdr_name_match = cdr_name
                    if cdr_idx >= len(cdr_regions):
                        if verbose:
                            print(f" warning:CDRindex {cdr_idx} (CDR: {len(cdr_regions)})")
                        break
                    
                    
                    remaining = binder_part[len(cdr_name):].strip('_').strip('-')
                    if len(remaining) == 1:
                        binder_aa = remaining.upper()
                    elif len(remaining) > 1:
                        
                        binder_aa = remaining[0].upper() if remaining[0].isalpha() else None
                    
                    if binder_aa is None:
                        if verbose:
                            print(f" warning:nofrom {binder_part} extractresidue")
                        break
                    
                    
                    cdr_start, cdr_end = cdr_regions[cdr_idx]
                    available_positions = [pos for pos in range(cdr_start, cdr_end) if pos not in seen_binder_positions]
                    if not available_positions:
                        if verbose:
                            print(f" warning:{cdr_name} noposition(already)")
                        break
                    
                    binder_idx = random.choice(available_positions)
                    if verbose:
                        print(f" {cdr_name}position: {binder_idx+1} (1-based) -> {binder_aa}")
                    break
            
            if cdr_name_match is None or binder_idx is None:
                if verbose:
                    print(f" warning:noparseCDR: {binder_part},support: CDR3_H, H_CDR3_H, L_CDR3_H")
                continue
        else:
            if verbose:
                print(f" warning:nobinder: {binder_part}")
            continue
        
        
        try:
            if receptor_part[-1].isalpha():
                receptor_pos_1based = int(receptor_part[:-1])
                receptor_aa = receptor_part[-1].upper()
            else:
                receptor_pos_1based = int(receptor_part)
                receptor_aa = None
            receptor_idx = receptor_pos_1based - 1  
        except ValueError:
            if verbose:
                print(f" warning:noparsereceptorposition: {receptor_part}")
            continue
        
        
        if receptor_aa is None:
            if receptor_idx >= 0 and receptor_idx < len(receptor_seq):
                receptor_aa = receptor_seq[receptor_idx].upper()
            else:
                if verbose:
                    print(f" warning:receptorposition {receptor_pos_1based} (1-based) ,skip")
                continue
        
        
        if binder_idx in seen_binder_positions:
            if verbose:
                print(f" error:Binderposition {binder_idx+1} (1-based) timesuse,vs！")
            continue
        
        if receptor_idx in seen_receptor_positions:
            if verbose:
                print(f" error:Receptorposition {receptor_pos_1based} (1-based) timesuse,vs！")
            continue
        
        
        fixed_residue_contacts[(binder_idx, receptor_idx)] = (binder_aa, receptor_aa)
        fixed_binder_positions.add(binder_idx)
        fixed_receptor_positions.add(receptor_idx)
        fixed_binder_aa_map[binder_idx] = binder_aa
        fixed_contact_pairs.append((receptor_idx, binder_idx))
        seen_binder_positions.add(binder_idx)
        seen_receptor_positions.add(receptor_idx)
        
        if verbose:
            print(f" fixedvs: Binder {binder_idx+1} ({binder_aa}) <-> Receptor {receptor_pos_1based} ({receptor_aa})")
    
    return fixed_residue_contacts, fixed_binder_positions, fixed_receptor_positions, fixed_binder_aa_map, fixed_contact_pairs


def randomize_cdr3_length(cdr_regions: list, binder_length: int, is_fab: bool = False, random_range: int = 3, min_cdr3_length: int = 10, max_cdr3_length: int = 25) -> list:
    """randomize cdr3 length."""
    import random
    
    modified_cdr_regions = cdr_regions.copy()
    
    if is_fab:
        
        return modified_cdr_regions
    else:
        
        
        
        if len(cdr_regions) >= 3:
            cdr3_start, cdr3_end = cdr_regions[2]
            cdr3_length = cdr3_end - cdr3_start
            
            
            
            new_cdr3_length = cdr3_length + random.randint(-random_range, random_range)
            
            
            new_cdr3_length = max(min_cdr3_length, min(new_cdr3_length, max_cdr3_length))
            
            
            max_allowed_length = binder_length - cdr3_start - 11  
            new_cdr3_length = min(new_cdr3_length, max_allowed_length)
            
            
            new_cdr3_length = max(new_cdr3_length, min_cdr3_length)
            
            new_cdr3_end = cdr3_start + new_cdr3_length
            
            
            if new_cdr3_end <= binder_length:
                modified_cdr_regions[2] = (cdr3_start, new_cdr3_end)
                print(f" CDR3length: {cdr3_length} -> {new_cdr3_length} (1-based: {cdr3_start+1}-{new_cdr3_end},: {min_cdr3_length}-{max_cdr3_length})")
            else:
                print(f" CDR3lengthfailed:computelength {new_cdr3_length} binderlength {binder_length}")
    
    return modified_cdr_regions



def get_fake_msa_dir_from_args(args):
    """get fake msa dir from args."""
    from pathlib import Path
    import tempfile
    
    
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif (hasattr(args, '_dream_config') and args._dream_config is not None
          and hasattr(args._dream_config, 'output') and args._dream_config.output.dir):
        output_dir = Path(args._dream_config.output.dir)
    
    
    if output_dir:
        msa_dir = output_dir / "fake_msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        return msa_dir
    else:
        
        return Path(tempfile.mkdtemp(prefix="fake_msa_"))


def parse_residue_ranges(range_str: str) -> list:
    """parse residue ranges."""
    residue_indices = []
    if not range_str or range_str.strip() == '':
        return residue_indices
    
    ranges = range_str.split(',')
    for r in ranges:
        r = r.strip()
        if '-' in r:
            start, end = r.split('-')
            start_idx = int(start.strip()) - 1  
            end_idx = int(end.strip())  
            residue_indices.extend(range(start_idx, end_idx))
        else:
            
            idx = int(r.strip()) - 1  
            residue_indices.append(idx)
    
    
    residue_indices = sorted(set(residue_indices))
    return residue_indices


def auto_pair_receptor_to_cdr(
    receptor_seq: str,
    receptor_indices: list,  
    cdr_regions: list,  
    binder_length: int,  
    exclude_binder_positions: set = None,  
    verbose: bool = True,
) -> tuple:
    """auto pair receptor to cdr."""
    import random
    
    
    POSITIVE_CHARGED = {"R", "K", "H"}
    NEGATIVE_CHARGED = {"D", "E"}
    POLAR_HYDROPHILIC = {"S", "T", "N", "Q"}
    HYDROPHOBIC = {"Y", "F", "W", "V", "I", "L", "M", "A", "C", "G", "P"}  
    
    def get_complementary_type(receptor_aa: str) -> list:
        """get complementary type."""
        receptor_aa = receptor_aa.upper()
        if receptor_aa in POSITIVE_CHARGED:
            
            return ["E", "D"]
        elif receptor_aa in NEGATIVE_CHARGED:
            
            return ["R", "K", "H"]
        elif receptor_aa in POLAR_HYDROPHILIC:
            
            return ["Y", "N", "Q"]
        elif receptor_aa in HYDROPHOBIC:
            
            return ["Y", "F", "L", "M", "I", "W", "V", "A"]
        else:
            
            return ["Y", "F", "L", "M", "I", "W", "V", "A"]
    
    
    receptor_residues = []
    for idx in receptor_indices:
        if idx < 0 or idx >= len(receptor_seq):
            if verbose:
                print(f"[WARNING] Receptorindex {idx} (sequencelength: {len(receptor_seq)}),skip")
            continue
        aa_type = receptor_seq[idx].upper()
        receptor_residues.append((idx, aa_type))
    
    if not receptor_residues:
        if verbose:
            print('[ERROR] noreceptorresidue')
        return []
    
    
    num_receptors = len(receptor_residues)
    num_cdrs = len(cdr_regions)
    
    
    per_cdr = 3  
    cdr_assignments = {}  # {cdr_idx: [receptor_idx, ...]}
    
    
    for i, (receptor_idx, aa_type) in enumerate(receptor_residues):
        
        cdr_idx = i % num_cdrs
        
        
        if cdr_idx not in cdr_assignments:
            cdr_assignments[cdr_idx] = []
        
        
        if len(cdr_assignments[cdr_idx]) < per_cdr:
            cdr_assignments[cdr_idx].append((receptor_idx, aa_type))
        else:
            
            found = False
            for j in range(num_cdrs):
                check_idx = (cdr_idx + j) % num_cdrs
                if check_idx not in cdr_assignments or len(cdr_assignments[check_idx]) < per_cdr:
                    if check_idx not in cdr_assignments:
                        cdr_assignments[check_idx] = []
                    cdr_assignments[check_idx].append((receptor_idx, aa_type))
                    found = True
                    break
            if not found:
                
                if num_cdrs > 0:
                    last_cdr = num_cdrs - 1
                    if last_cdr not in cdr_assignments:
                        cdr_assignments[last_cdr] = []
                    cdr_assignments[last_cdr].append((receptor_idx, aa_type))
    
    if verbose:
        print(f"\n[autovs] result:")
        for cdr_idx, assignments in cdr_assignments.items():
            cdr_start, cdr_end = cdr_regions[cdr_idx]
            print(f"  CDR{cdr_idx+1} [{cdr_start}, {cdr_end}): {len(assignments)} receptorresidue")
            for receptor_idx, aa_type in assignments:
                print(f" Receptorresidue {receptor_idx+1} ({aa_type})")
    
    
    if exclude_binder_positions is None:
        exclude_binder_positions = set()
    used_binder_positions = set(exclude_binder_positions)  
    
    if verbose and exclude_binder_positions:
        print(f"\n binderposition(fixedresidue): {sorted([idx + 1 for idx in exclude_binder_positions])} (1-based)")
    
    
    contact_pairs = []
    binder_aa_map = {}  
    
    for cdr_idx, assignments in cdr_assignments.items():
        cdr_start, cdr_end = cdr_regions[cdr_idx]
        cdr_length = cdr_end - cdr_start
        
        for receptor_idx, receptor_aa in assignments:
            
            complementary_types = get_complementary_type(receptor_aa)
            selected_type = random.choice(complementary_types)
            
            
            available_positions = [pos for pos in range(cdr_start, cdr_end) if pos not in used_binder_positions]
            if not available_positions:
                
                available_positions = [pos for pos in range(binder_length) if pos not in used_binder_positions]
                if not available_positions:
                    if verbose:
                        print(f"[WARNING] noasreceptorresidue {receptor_idx+1} tobinderposition(asfixedresidue)")
                    continue
            
            binder_pos = random.choice(available_positions)
            used_binder_positions.add(binder_pos)
            
            contact_pairs.append((receptor_idx, binder_pos))
            binder_aa_map[binder_pos] = selected_type  
            
            if verbose:
                print(f" vs: Receptor {receptor_idx+1} ({receptor_aa}) -> Binder {binder_pos+1} (need: {selected_type})")
    
    return contact_pairs, binder_aa_map


def prepare_contacts_and_fake_msa(
    receptor_seq: str,
    receptor_seq_original: str,  
    binder_length: int,
    binder_seq_template: str,
    cdr_regions: list,
    nanobody_mode: bool,
    fab_mode: bool,
    args,
    fixed_residue_contacts_str: str = None,
    required_contacts_str: str = None,
    max_msa_seqs: int = 700,
    verbose: bool = True,
) -> tuple:
    """prepare contacts and fake msa."""
    from dream_boltz2.msa.fake_msa import generate_fake_msa_mode2
    import tempfile
    from pathlib import Path
    
    
    receptor_seq_processed = receptor_seq
    original_receptor_res_names = None
    
    if args.anti_contact_residues:
        mutation_indices = parse_residue_ranges(args.anti_contact_residues)
        valid_indices = [idx for idx in mutation_indices if 0 <= idx < len(receptor_seq_original)]
        if valid_indices:
            from boltz.data.const import prot_letter_to_token
            original_receptor_res_names = []
            for idx, aa in enumerate(receptor_seq_original):
                if idx in valid_indices:
                    original_receptor_res_names.append(prot_letter_to_token.get(aa, 'UNK'))
                else:
                    original_receptor_res_names.append(prot_letter_to_token.get(aa, 'UNK'))
            receptor_seq_processed = apply_anti_contact_mutation(
                receptor_seq_original, valid_indices, args.anti_contact_mutation_type
            )
    
    
    fixed_residue_contacts = None
    fixed_binder_positions = set()
    fixed_receptor_positions = set()
    fixed_binder_aa_map = {}
    fixed_contact_pairs = []
    
    if fixed_residue_contacts_str:
        if verbose:
            print(f"\n{'='*60}")
            print(f" parseFixed Residue Contacts(fixedresiduevs)")
            print(f"{'='*60}")
        
        fixed_residue_contacts, fixed_binder_positions, fixed_receptor_positions, fixed_binder_aa_map, fixed_contact_pairs = parse_fixed_residue_contacts(
            fixed_residue_contacts_str,
            receptor_seq_processed,
            cdr_regions,
            binder_length,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n fixedresidue:")
            print(f" fixedvs: {len(fixed_residue_contacts)}")
            print(f" fixedbinderposition: {sorted([idx + 1 for idx in fixed_binder_positions])} (1-based)")
            print(f" fixedreceptorposition: {sorted([idx + 1 for idx in fixed_receptor_positions])} (1-based)")
    
    
    auto_generated_contact_pairs = None
    binder_aa_map = {}
    
    if required_contacts_str:
        receptor_positions_str = required_contacts_str.split(',')
        receptor_indices_raw = [int(pos.strip()) - 1 for pos in receptor_positions_str if pos.strip()]
        
        
        receptor_indices = [idx for idx in receptor_indices_raw if idx not in fixed_receptor_positions]
        
        if receptor_indices and cdr_regions:
            auto_generated_contact_pairs, binder_aa_map = auto_pair_receptor_to_cdr(
                receptor_seq=receptor_seq_processed,
                receptor_indices=receptor_indices,
                cdr_regions=cdr_regions,
                binder_length=len(binder_seq_template),
                exclude_binder_positions=fixed_binder_positions,
                verbose=verbose
            )
    
    
    binder_seq_for_yaml = list(binder_seq_template)
    
    
    if fixed_binder_aa_map:
        for binder_idx, aa_type in fixed_binder_aa_map.items():
            if 0 <= binder_idx < len(binder_seq_for_yaml):
                binder_seq_for_yaml[binder_idx] = aa_type
    
    
    for binder_idx, aa_type in binder_aa_map.items():
        if 0 <= binder_idx < len(binder_seq_for_yaml):
            binder_seq_for_yaml[binder_idx] = aa_type
    
    binder_seq_for_yaml_str = ''.join(binder_seq_for_yaml)
    
    
    binder_seq_for_msa = list(binder_seq_for_yaml_str)
    for i in range(len(binder_seq_for_msa)):
        if binder_seq_for_msa[i] == 'X':
            binder_seq_for_msa[i] = 'A'
    binder_seq_for_msa_str = ''.join(binder_seq_for_msa)
    
    binder_template_modified = binder_seq_for_yaml_str
    
    
    receptor_msa_file_for_fake = None  
    binder_msa_file_for_fake = None  
    
    
    all_contact_pairs = fixed_contact_pairs.copy()
    if auto_generated_contact_pairs:
        all_contact_pairs.extend(auto_generated_contact_pairs)
    
    if all_contact_pairs:
        try:
            
            temp_msa_dir = get_fake_msa_dir_from_args(args)
            if "fake_msa" in str(temp_msa_dir):
                if verbose:
                    print(f" Fake MSA saveto: {temp_msa_dir.absolute()}")
            
            receptor_df, binder_df = generate_fake_msa_mode2(
                receptor_seq=receptor_seq_processed,
                binder_seq=binder_seq_for_msa_str,
                contact_pairs=all_contact_pairs,
                mapping_mode="complementary",
                num_seqs=max_msa_seqs,
                correlation_strength=0.9,
                output_dir=temp_msa_dir,
                verbose=verbose
            )
            
            receptor_msa_path = temp_msa_dir / "receptor_msa.csv"
            binder_msa_path = temp_msa_dir / "binder_msa.csv"
            receptor_df.to_csv(receptor_msa_path, index=False)
            binder_df.to_csv(binder_msa_path, index=False)
            
            
            receptor_msa_file_for_fake = str(receptor_msa_path)
            binder_msa_file_for_fake = str(binder_msa_path)
            if verbose:
                print(f" Fake MSAalreadygenerate(vsMSA):")
                print(f"   Receptor MSA: {receptor_msa_file_for_fake}")
                print(f"   Binder MSA: {binder_msa_file_for_fake}")
                print(f" requiredusevsreceptorandbinder MSAkeepvs！")
        except Exception as e:
            if verbose:
                print(f" generatefake MSAfailed: {e}")
                import traceback
                traceback.print_exc()
    
    
    from dream_boltz2.features.complex import FeaturePreparatorComplex
    
    feature_prep = FeaturePreparatorComplex(
        boltz_model=None,  
        device=args.device,
        mode='complex'
    )
    
    
    
    
    return {
        'binder_template_modified': binder_template_modified,
        'receptor_msa_file_for_fake': receptor_msa_file_for_fake,  
        'binder_msa_file_for_fake': binder_msa_file_for_fake,
        'fixed_binder_positions': fixed_binder_positions,
        'fixed_receptor_positions': fixed_receptor_positions,
        'receptor_seq_processed': receptor_seq_processed,
        'original_receptor_res_names': original_receptor_res_names,
    }


def apply_anti_contact_mutation(sequence: str, residue_indices: list, mutation_type: str = 'N') -> str:
    """apply anti contact mutation."""
    mutated_seq = list(sequence)
    for idx in residue_indices:
        if 0 <= idx < len(mutated_seq):
            mutated_seq[idx] = mutation_type
    
    return ''.join(mutated_seq)


def parse_cdr_regions(cdr_str: str, binder_length: int, is_fab: bool = False) -> list:
    """parse cdr regions."""
    if cdr_str.lower() == 'fab':
        
        if is_fab:
            _, cdr_regions, _ = get_fab_template('fab_default')
            return cdr_regions
        else:
            raise ValueError("Fab CDR regions require --fab flag")
    elif cdr_str.lower() == 'nanobody':
        
        
        
        if binder_length >= 120:
            
            
            return [
                (25, 33),   
                (50, 58),   
                (96, 115),  
            ]
        else:
            
            scale = binder_length / 120.0
            return [
                (int(25 * scale), int(34 * scale)),
                (int(50 * scale), int(59 * scale)),
                (int(97 * scale), int(118 * scale)),
            ]
    else:
        
        cdr_regions = []
        for region_str in cdr_str.split(','):
            parts = region_str.strip().split(':')
            if len(parts) == 2:
                start = int(parts[0]) - 1  
                end = int(parts[1])  
                cdr_regions.append((start, end))
            else:
                raise ValueError(f"Invalid CDR region format: {region_str}. Expected 'start:end'")
        return cdr_regions


def cdr_regions_for_pdb_from_metadata(
    meta: Optional[Dict],
    fallback_binder_relative: Optional[List[Tuple[int, int]]],
) -> Optional[List[Tuple[int, int]]]:
    """cdr regions for pdb from metadata."""
    if meta is None:
        return fallback_binder_relative
    bs = meta.get('binder_slice')
    og = meta.get('optimizable_regions')
    if not og or bs is None:
        return fallback_binder_relative
    out: List[Tuple[int, int]] = []
    try:
        for item in og:
            if isinstance(item, slice):
                g0, g1 = item.start, item.stop
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                g0, g1 = int(item[0]), int(item[1])
            else:
                return fallback_binder_relative
            out.append((g0 - bs.start, g1 - bs.start))
        return out if out else fallback_binder_relative
    except (TypeError, ValueError):
        return fallback_binder_relative


def create_region_mask(
    sequence_length: int,
    regions: list,  
    device: str = 'cuda'
) -> torch.Tensor:
    """create region mask."""
    mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    
    for region in regions:
        if isinstance(region, slice):
            mask[region] = True
        elif isinstance(region, tuple) and len(region) == 2:
            start, end = region
            mask[start:end] = True
        else:
            raise ValueError(f"Invalid region format: {region}. Expected slice or (start, end) tuple.")
    
    return mask


def scale_pair_representation(
    z: torch.Tensor,  # [B, L, L, D]
    creativity: float,
    binder_mask: torch.Tensor,  # [L] bool
    receptor_mask: torch.Tensor,  # [L] bool
    cdr_mask: Optional[torch.Tensor] = None,  
    hotspot_indices: Optional[list] = None,  
    receptor_start: int = 0,  
) -> torch.Tensor:
    """Scale pair rows by (1 - creativity). Receptor-receptor pairs and hotspot anchors stay at 1."""
    if creativity <= 0.0:
        return z  
    
    scale = 1.0 - creativity  
    L = z.shape[1]
    device = z.device
    
    
    hotspot_mask = torch.zeros(L, dtype=torch.bool, device=device)
    if hotspot_indices is not None and len(hotspot_indices) > 0:
        
        global_hotspot_indices = [receptor_start + idx for idx in hotspot_indices]
        for idx in global_hotspot_indices:
            if 0 <= idx < L:
                hotspot_mask[idx] = True
    
    
    scaling_mask = torch.ones(L, L, device=device)
    
    if cdr_mask is not None:
        
        cdr_mask = cdr_mask.bool()
        framework_mask = binder_mask & (~cdr_mask)
        
        
        non_hotspot_receptor = receptor_mask & (~hotspot_mask)
        
        
        cdr_2d = cdr_mask[:, None] & cdr_mask[None, :]
        scaling_mask[cdr_2d] = scale
        
        
        cdr_non_hotspot = cdr_mask[:, None] & non_hotspot_receptor[None, :]
        non_hotspot_cdr = non_hotspot_receptor[:, None] & cdr_mask[None, :]
        scaling_mask[cdr_non_hotspot] = scale
        scaling_mask[non_hotspot_cdr] = scale
        
        
        
        
        
        framework_receptor = framework_mask[:, None] & receptor_mask[None, :]
        receptor_framework = receptor_mask[:, None] & framework_mask[None, :]
        scaling_mask[framework_receptor] = scale
        scaling_mask[receptor_framework] = scale
        
        
    else:
        
        
        non_hotspot_receptor = receptor_mask & (~hotspot_mask)
        
        
        binder_2d = binder_mask[:, None] & binder_mask[None, :]
        scaling_mask[binder_2d] = scale
        
        
        binder_non_hotspot = binder_mask[:, None] & non_hotspot_receptor[None, :]
        non_hotspot_binder = non_hotspot_receptor[:, None] & binder_mask[None, :]
        scaling_mask[binder_non_hotspot] = scale
        scaling_mask[non_hotspot_binder] = scale
        
        
        
    
    
    scaling_mask = scaling_mask.view(1, L, L, 1)
    z_scaled = z * scaling_mask
    
    return z_scaled


def initialize_s_z(
    pairformer: PairformerWrapper,  
    feats: dict,
    metadata: dict,
    optimizable_regions: list = None,  
    device: str = 'cuda',
    
    creativity: float = 0.0,
    cdr_mask: Optional[torch.Tensor] = None,
    receptor_mask: Optional[torch.Tensor] = None,
    binder_mask: Optional[torch.Tensor] = None,
    hotspot_indices: Optional[list] = None,
    receptor_start: int = 0,
    
    scaling_residues: Optional[List[int]] = None,  
    scaling_residues_only: bool = False,  
) -> tuple[list, callable]:
    """initialize s z."""
    
    feats_with_metadata = feats.copy()
    feats_with_metadata['metadata'] = metadata
    
    
    
    with torch.no_grad():
        s_init, z_init = pairformer(
            feats=feats_with_metadata,
            return_s_inputs=False,
            
            creativity=creativity,
            cdr_mask=cdr_mask,
            receptor_mask=receptor_mask,
            binder_mask=binder_mask,
            hotspot_indices=hotspot_indices,
            receptor_start=receptor_start,
            scaling_residues=scaling_residues,  
            scaling_residues_only=scaling_residues_only,  
        )
        # s_init: [1, L, D_s], z_init: [1, L, L, D_z]
    
    L = s_init.shape[1]  
    
    
    receptor_slice = metadata.get('receptor_slice', slice(0, 0))
    binder_slice = metadata.get('binder_slice')
    
    if binder_slice is None:
        raise ValueError("metadata must contain 'binder_slice'")
    
    
    if optimizable_regions is not None:
        
        print('\n[init] argument(regionmode):extractregionasoptimizationargument...')
        region_mask = create_region_mask(L, optimizable_regions, device)
        print(f" sequencelength: {L}")
        print(f" optimizationregion: {region_mask.sum().item()}")
        print(f" optimizationregion: {optimizable_regions}")
    else:
        
        print('\n[init] argument(bindermode):extractbinderregionasoptimizationargument...')
        region_mask = torch.zeros(L, dtype=torch.bool, device=device)
        if binder_slice.stop > 0:
            region_mask[binder_slice] = True
        print(f" sequencelength: {L}")
        print(f" optimizationregion: {region_mask.sum().item()} (binder)")
    
    
    receptor_mask = torch.zeros(L, dtype=torch.bool, device=device)
    if receptor_slice.stop > 0:
        receptor_mask[receptor_slice] = True
    
    
    framework_mask = ~region_mask & ~receptor_mask
    
    print(f" receptorregion: {receptor_mask.sum().item()}")
    if optimizable_regions is not None:
        print(f" fixedregion: {framework_mask.sum().item()} (binderinoptimizationregion)")
    else:
        print(f" fixedregion: {framework_mask.sum().item()} (no,asbinderoptimization)")
    
    print(f" s shape: {s_init.shape}")
    print(f" z shape: {z_init.shape}")
    
    # ============================================================
    
    # ============================================================
    s_template = s_init.detach().clone()
    s_params = s_init[:, region_mask, :].detach().clone().requires_grad_(True)
    
    print(f"\n sargument:")
    print(f": {L}, optimizationargument: {s_params.shape[1]} (optimizationregion)")
    if optimizable_regions is not None:
        print(f" fixedregionandreceptornotandoptimization(computeinnotcontains)")
    else:
        print(f" receptornotandoptimization(computeinnotcontains)")
    
    # ============================================================
    
    # ============================================================
    z_template = z_init.detach().clone()
    
    
    if 'token_bonds' in feats and feats['token_bonds'] is not None:
        token_bonds_feat = feats['token_bonds']  # [1, L, L, 1]
        if isinstance(token_bonds_feat, torch.Tensor):
            bond_count = (token_bonds_feat > 0).sum().item()
            print(f"\n [Z_INIT DEBUG] checkz_initintoken_bondsinfo:")
            print(f" token_bondsinfeatsin:, shape={token_bonds_feat.shape}, bonds={bond_count}")
            
            if bond_count > 0:
                bond_mask = (token_bonds_feat.squeeze(-1) > 0)  # [1, L, L]
                no_bond_mask = ~bond_mask
                if no_bond_mask.sum() > 0:
                    z_init_with_bond = z_init[bond_mask.unsqueeze(-1).expand_as(z_init)]
                    z_init_without_bond = z_init[no_bond_mask.unsqueeze(-1).expand_as(z_init)]
                    mean_diff = (z_init_with_bond.mean() - z_init_without_bond.mean()).abs().item()
                    print(f" z_initinbondposition: {z_init_with_bond.mean().item():.4f}")
                    print(f" z_initinnobondposition: {z_init_without_bond.mean().item():.4f}")
                    print(f": {mean_diff:.4f} (if>0.1,token_bondsinfoalready)")
    
    
    
    
    
    
    mask_2d = region_mask.view(1, L, 1).to(device)  # [1, L, 1]
    receptor_mask_2d = receptor_mask.view(1, L, 1).to(device)  # [1, L, 1]
    
    
    framework_mask = ~region_mask & ~receptor_mask
    framework_mask_2d = framework_mask.view(1, L, 1).to(device)  # [1, L, 1]
    
    
    
    
    ligand_mask = torch.zeros(L, dtype=torch.bool, device=device)
    if 'mol_type' in feats and feats['mol_type'] is not None:
        mol_type = feats['mol_type']  
        if mol_type.dim() == 2:
            mol_type = mol_type.squeeze(0)  # [L]
        ligand_mask = (mol_type != 0).to(device)  # ligand tokens
        if ligand_mask.sum() > 0:
            print(f" to {ligand_mask.sum().item()} ligand tokens: {ligand_mask.nonzero(as_tuple=False).squeeze(-1).tolist()}")
    
    ligand_mask_2d = ligand_mask.view(1, L, 1).to(device)  # [1, L, 1]
    
    
    ligand_ligand_pairs = (ligand_mask_2d & ligand_mask_2d.transpose(1, 2))  # [1, L, L]
    
    
    bond_constraint_pairs = torch.zeros(1, L, L, dtype=torch.bool, device=device)
    if 'token_bonds' in feats and feats['token_bonds'] is not None:
        token_bonds_feat = feats['token_bonds']  
        if token_bonds_feat.dim() == 4:
            token_bonds_feat = token_bonds_feat.squeeze(0).squeeze(-1)  # [L, L]
        elif token_bonds_feat.dim() == 3:
            token_bonds_feat = token_bonds_feat.squeeze(-1)  # [L, L]
        
        bond_constraint_pairs = (token_bonds_feat > 0).unsqueeze(0)  # [1, L, L]
        bond_count = bond_constraint_pairs.sum().item()
        if bond_count > 0:
            print(f" to {bond_count} bondvs(ligandandchainbondconstraint)")
            
            if ligand_mask.sum() > 0:
                ligand_internal_bonds = bond_constraint_pairs & ligand_ligand_pairs
                ligand_cross_bonds = bond_constraint_pairs & ~ligand_ligand_pairs
                ligand_internal_count = ligand_internal_bonds.sum().item()
                ligand_cross_count = ligand_cross_bonds.sum().item()
                print(f" - Ligandbonds: {ligand_internal_count} (protectedligand)")
                print(f" - chainbondconstraint: {ligand_cross_count} (protectedbondconstraint)")
    
    
    
    
    
    
    protected_pairs = ligand_ligand_pairs | bond_constraint_pairs  # [1, L, L]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # 
    
    
    
    
    receptor_both = (receptor_mask_2d & receptor_mask_2d.transpose(1, 2))  
    framework_both = (framework_mask_2d & framework_mask_2d.transpose(1, 2))  
    both_fixed = receptor_both | framework_both  
    
    
    base_mask = (mask_2d | mask_2d.transpose(1, 2))  # [1, L, L]
    
    
    
    framework_receptor_pairs = (framework_mask_2d & receptor_mask_2d.transpose(1, 2)) | \
                               (receptor_mask_2d & framework_mask_2d.transpose(1, 2))  
    
    
    
    
    
    
    
    z_mask_bool = (base_mask | framework_receptor_pairs) & ~both_fixed & ~protected_pairs  # [1, L, L]
    
    
    mask_expanded = z_mask_bool.unsqueeze(-1).expand_as(z_init)  # [1, L, L, D_z]
    
    
    if protected_pairs.sum() > 0:
        protected_positions = protected_pairs.squeeze(0).nonzero(as_tuple=False)  # [num_protected, 2]
        if protected_positions.shape[0] > 0:
            print(f"\n [PROTECTED PAIRS DEBUG] to {protected_positions.shape[0]} needprotectedvs")
            
            protected_in_mask_count = 0
            protected_not_in_mask_count = 0
            for pos in protected_positions[:5]:  
                i, j = pos[0].item(), pos[1].item()
                in_mask = mask_expanded[0, i, j, 0].item() > 0
                is_ligand_pair = ligand_ligand_pairs[0, i, j].item() > 0
                is_bond_pair = bond_constraint_pairs[0, i, j].item() > 0
                pair_type = "ligand-ligand" if is_ligand_pair else ('bondconstraint' if is_bond_pair else '')
                if in_mask:
                    protected_in_mask_count += 1
                    print(f"    [{i}, {j}] ({pair_type}) inoptimizationmaskin (notthis！)")
                else:
                    protected_not_in_mask_count += 1
                    print(f"    [{i}, {j}] ({pair_type}) notinoptimizationmaskin (usez_template)")
            if protected_positions.shape[0] > 5:
                print(f"... ( {protected_positions.shape[0] - 5} vs)")
            print(f": {protected_in_mask_count} vsinmaskin(notthis), {protected_not_in_mask_count} vsnotinmaskin()")
    
    
    z_params_flat = z_init[mask_expanded].detach().clone().requires_grad_(True)
    
    num_optimizable_pairs = z_params_flat.shape[0] // z_init.shape[-1]
    
    
    
    is_fab_mode = optimizable_regions is not None and len(optimizable_regions) == 6
    
    if is_fab_mode:
        
        
        print(f"\n zargument (Fabmode - chain):")
        print(f" optimizationoptimizationregion {num_optimizable_pairs} vs")
        print(f"     ")
        print(f" Fab(s,z):")
        print(f" s: [1, L, D_s],inL = receptorlength + Hchainlength + Lchainlength")
        print(f" z: [1, L, L, D_z],containsallresiduevs")
        print(f"     ")
        print(f" zcontainsvs:")
        print(f" 1. receptor-receptor pair(fixed,notoptimization)")
        print(f" 2. receptor-Hchainvs(receptor-H-CDRandreceptor-H-framework)")
        print(f" 3. receptor-Lchainvs(receptor-L-CDRandreceptor-L-framework)")
        print(f" 4. Hchain-Hchainvs(chain:H-CDR-H-CDR, H-CDR-H-framework, H-framework-H-framework)")
        print(f" 5. Hchain-Lchainvs(chain:H-CDR-L-CDR, H-CDR-L-framework, H-framework-L-CDR, H-framework-L-framework)")
        print(f" 6. Lchain-Lchainvs(chain:L-CDR-L-CDR, L-CDR-L-framework, L-framework-L-framework)")
        print(f"     ")
        print(f" gradientvs(optimizationCDRvs):")
        print(f" contains(gradient):")
        print(f" - H-CDR-H-CDRvs(HchainCDR)")
        print(f" - L-CDR-L-CDRvs(LchainCDR)")
        print(f" - H-CDR-L-CDRvs(HchainandLchainCDR,chainCDR)")
        print(f" - H-CDR-receptor pair(HchainCDRandreceptor)")
        print(f" - L-CDR-receptor pair(LchainCDRandreceptor)")
        print(f" - H-CDR-H-frameworkvs(HchainCDRandHchainframework)")
        print(f" - L-CDR-L-frameworkvs(LchainCDRandLchainframework)")
        print(f" - H-CDR-L-frameworkvs(HchainCDRandLchainframework)")
        print(f" - L-CDR-H-frameworkvs(LchainCDRandHchainframework)")
        print(f" (nogradient):")
        print(f" - H-framework-H-frameworkvs(fixed)")
        print(f" - L-framework-L-frameworkvs(fixed)")
        print(f" - H-framework-L-frameworkvs(fixed)")
        print(f" - receptor-receptor pair(fixed)")
    else:
        print(f"\n zargument:")
        print(f" optimizationoptimizationregion {num_optimizable_pairs} vs")
        print(f":optimization-optimization, optimization-fixedregion, optimization-receptor pair(optimizationregionin,needoptimization！)")
        print(f":")
        print(f" - receptor-receptor pair(fixed)")
        print(f" - framework-frameworkvs(fixed)")
        if protected_pairs.sum() > 0:
            protected_count = protected_pairs.sum().item()
            ligand_ligand_count = ligand_ligand_pairs.sum().item()
            bond_count = bond_constraint_pairs.sum().item()
            print(f" - needprotectedvs({protected_count}):")
            print(f" * Ligand-ligandvs({ligand_ligand_count},protectedligand)")
            print(f" * Bondconstraintvs({bond_count},protectedbondconstraint)")
        print(f" note:ligand-residuebondvscanoptimization(ifgradient)")
    
    # ============================================================
    
    # ============================================================
    def assemble_full_sz():
        """assemble full sz."""
        
        s_full = s_template.clone()  
        s_full[:, region_mask, :] = s_params  
        
        
        
        z_delta = torch.zeros_like(z_template)  
        
        
        mask_indices = mask_expanded.nonzero(as_tuple=False)  # [num_masked, 4] (batch, i, j, d)
        if mask_indices.shape[0] > 0:
            
            z_delta = z_delta.index_put_(
                tuple(mask_indices[:, dim].t() for dim in range(4)),
                z_params_flat.view(-1),
                accumulate=False
            )
        
        z_full = torch.where(mask_expanded, z_delta, z_template)  
        
        
        
        if protected_pairs.sum() > 0 and not hasattr(assemble_full_sz, '_protected_verify_printed'):
            
            protected_not_in_opt_mask = protected_pairs & ~mask_expanded[:, :, :, 0]
            if protected_not_in_opt_mask.sum() > 0:
                
                z_full_protected_values = z_full[protected_not_in_opt_mask.unsqueeze(-1).expand_as(z_full)]
                z_template_protected_values = z_template[protected_not_in_opt_mask.unsqueeze(-1).expand_as(z_template)]
                if z_full_protected_values.numel() > 0 and z_template_protected_values.numel() > 0:
                    max_diff = (z_full_protected_values - z_template_protected_values).abs().max().item()
                    if max_diff < 1e-6:
                        print(f" [PROTECTED PAIRS] z_full needprotectedvsinfo (max_diff={max_diff:.2e})")
                    else:
                        print(f" [PROTECTED PAIRS] z_full and z_template inprotectedposition (max_diff={max_diff:.2e})")
                else:
                    print(f" [PROTECTED PAIRS] protectedpositionasempty,skipcheck")
                assemble_full_sz._protected_verify_printed = True
        
        return s_full, z_full
    
    return [s_params, z_params_flat], assemble_full_sz


def compute_structure_loss(
    pred_coords: torch.Tensor,  # [1, N_atoms, 3]
    target_coords: Optional[torch.Tensor] = None,  
    feats: dict = None,
    metadata: dict = None,
    
    use_rg_loss: bool = True,
    use_helix_loss: bool = True,
    use_hotspot_loss: bool = True,
    
    rg_weight: float = 1.0,
    helix_weight: float = 0.5,
    hotspot_weight: float = 0.3,
    
    hotspot_indices: Optional[list] = None,  
) -> Tuple[torch.Tensor, dict]:
    """compute structure loss."""
    from dream_boltz2.losses.coords_losses import RGLoss, HelixLossCoords
    from dream_boltz2.losses.interface_losses import HotspotResidueLoss
    
    device = pred_coords.device
    loss_info = {}
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    
    if target_coords is not None:
        mse_loss = torch.mean((pred_coords - target_coords) ** 2)
        loss_info['mse_loss'] = mse_loss.item()
        return mse_loss, loss_info
    
    # ============================================================
    
    # ============================================================
    if use_rg_loss:
        try:
            rg_loss_fn = RGLoss(
                weight=rg_weight,
                target_mode='fixed',
                target_rg=10.0,
                tolerance=1.5
            )
            rg_loss, rg_info = rg_loss_fn(
                coords=pred_coords,
                feats=feats,
                metadata=metadata
            )
            total_loss = total_loss + rg_loss
            loss_info['rg_loss'] = rg_info
        except Exception as e:
            print(f"[WARNING] RG Loss failed: {e}")
            loss_info['rg_loss'] = {'error': str(e)}
    
    # ============================================================
    
    # ============================================================
    if use_helix_loss:
        try:
            helix_loss_fn = HelixLossCoords(
                weight=helix_weight,
                offset=3,  
                target_distance=5.5
            )
            helix_loss, helix_info = helix_loss_fn(
                coords=pred_coords,
                feats=feats,
                metadata=metadata
            )
            total_loss = total_loss + helix_loss
            loss_info['helix_loss'] = helix_info
        except Exception as e:
            print(f"[WARNING] Helix Loss failed: {e}")
            loss_info['helix_loss'] = {'error': str(e)}
    
    # ============================================================
    
    # ============================================================
    if use_hotspot_loss and hotspot_indices is not None:
        try:
            hotspot_loss_fn = HotspotResidueLoss(
                hotspot_indices=hotspot_indices,
                target_distance=8.0,  
                weight=hotspot_weight
            )
            hotspot_loss, hotspot_info = hotspot_loss_fn(
                coords=pred_coords,
                feats=feats,
                metadata=metadata
            )
            total_loss = total_loss + hotspot_loss
            loss_info['hotspot_loss'] = hotspot_info
        except Exception as e:
            print(f"[WARNING] Hotspot Loss failed: {e}")
            import traceback
            traceback.print_exc()
            loss_info['hotspot_loss'] = {'error': str(e)}
    elif use_hotspot_loss:
        if hotspot_indices is None:
            print(f"[WARNING] Hotspot Loss requires hotspot_indices, skipping")
    
    
    if total_loss.item() == 0.0:
        print(f"[WARNING] No loss computed, using fallback")
        total_loss = torch.mean(pred_coords ** 2)
        loss_info['fallback_loss'] = total_loss.item()
    
    return total_loss, loss_info


def optimize_s_z(
    boltz_model: Boltz2,
    pairformer: PairformerWrapper,  
    diffusion: DiffusionWrapper,
    feats: dict,
    metadata: dict,
    num_steps: int = 200,
    lr: float = 0.2,
    device: str = 'cuda',
    target_coords: Optional[torch.Tensor] = None,
    
    use_rg_loss: bool = True,
    use_helix_loss: bool = True,
    use_hotspot_loss: bool = True,
    rg_weight: float = 1.0,
    helix_weight: float = 0.5,
    hotspot_weight: float = 0.3,
    hotspot_indices: Optional[list] = None,  
    
    use_distogram_penalty: bool = False,
    distogram_penalty_weight: float = 1.0,
    distogram_penalty_regions: Optional[list] = None,  
    distogram_penalty_steps: int = 0,  
    
    optimizable_regions: Optional[list] = None,  
    
    early_filter_threshold: float = 1.0,  
    
    creativity: float = 0.0,  
    
    scaling_residues: Optional[List[int]] = None,  
    scaling_residues_only: bool = False,  
):
    """optimize s z."""
    print("\n" + "="*80)
    print('optimization(s,z)')
    print("="*80)
    
    
    
    
    
    print(f"\n useargument(regionoptimization)")
    if optimizable_regions is not None:
        print(f" mode: region(optimizationregion,CDRregion)")
        print(f" optimizationregion: {optimizable_regions}")
    else:
        print(f" mode: binder(optimizationbinder)")
    print(f" framework: BinderandNanobodyuseargument")
    
    
    binder_slice = metadata.get('binder_slice')
    receptor_slice = metadata.get('receptor_slice')
    L = feats['token_pad_mask'].shape[1]
    
    
    nanobody_mode = optimizable_regions is not None and len(optimizable_regions) == 3
    fab_mode = optimizable_regions is not None and len(optimizable_regions) == 6
    
    
    binder_mask = torch.zeros(L, dtype=torch.bool, device=device)
    receptor_mask = torch.zeros(L, dtype=torch.bool, device=device)
    cdr_mask = None
    
    if binder_slice is not None:
        binder_mask[binder_slice] = True
    if receptor_slice is not None:
        receptor_mask[receptor_slice] = True
    
    if creativity > 0.0:
        print(f"\n alphamode(round recycling in！)")
        print(f" alphafactor: {creativity:.2f}")
        
        if nanobody_mode or fab_mode:
            cdr_mask = torch.zeros(L, dtype=torch.bool, device=device)
            for region in optimizable_regions:
                if isinstance(region, slice):
                    cdr_mask[region] = True
                elif isinstance(region, tuple) and len(region) == 2:
                    cdr_mask[region[0]:region[1]] = True
            print(f" mode: {'Nanobody' if nanobody_mode else 'Fab'}(scaling CDR Z_ij)")
            print(f" CDR residue: {cdr_mask.sum().item()}")
        else:
            print(f" mode: ")
            print(f" scaling:")
            print(f" binder-binder: scaling")
            print(f" binder-hotspotreceptor: scaling")
            print(f" binder-hotspotresidue: unscaled(keep)")
            print(f" receptor-receptor: unscaled")
        
        print(f" Binder residue: {binder_mask.sum().item()}")
        print(f" Receptor residue: {receptor_mask.sum().item()}")
        if hotspot_indices:
            print(f" hotspotresidue: {len(hotspot_indices)}(binder andhotspotresidue Z_ij unscaled,keep)")
    
    params, assemble_full_sz = initialize_s_z(
        pairformer, feats, metadata, optimizable_regions, device,
        
        creativity=creativity,
        cdr_mask=cdr_mask,
        receptor_mask=receptor_mask,
        binder_mask=binder_mask,
        hotspot_indices=hotspot_indices,
        receptor_start=receptor_slice.start if receptor_slice else 0,
        
        scaling_residues=scaling_residues,
        scaling_residues_only=scaling_residues_only,
    )
    s_params, z_params_flat = params
    
    
    
    s_init, z_init = assemble_full_sz()
    s_init = s_init.detach().clone()
    z_init = z_init.detach().clone()
    print(f"\n alreadysave(s,z)(sequence),forPDBgenerate")
    
    
    optimizer = optim.Adam(params, lr=lr)
    
    
    monitor_memory = device == 'cuda' and torch.cuda.is_available()
    if monitor_memory:
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        print(f"\n:")
        print(f": {initial_memory:.2f} GB")
    else:
        initial_memory = 0.0
        monitor_memory = False
    
    print(f"\noptimizationconfig:")
    print(f": {lr}")
    print(f" optimizationstep: {num_steps}")
    print(f" optimizationargument: s_params ({s_params.shape}), z_params_flat ({z_params_flat.shape})")
    if optimizable_regions is not None:
        print(f" mode: regionoptimization(optimizationregion,regionfixed)")
    else:
        print(f" mode: binderoptimization(optimizationbinder,receptorfixed)")
    print(f": gradient,optimizationnottofixedregionandreceptor")
    print(f" Sigma: 250.0 (fixed,vsBoltz2in15.625,78%)")
    print(f" note: sigma=250X0_hat,gradient")
    print(f" notautoupdate(250 > 100)")
    
    
    
    
    
    feats_detached = {}
    receptor_slice = metadata.get('receptor_slice', slice(0, 0))
    binder_slice = metadata.get('binder_slice')
    
    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            
            
            
            feats_detached[k] = v.detach() if v.requires_grad else v
        else:
            feats_detached[k] = v
    
    
    
    with torch.no_grad():
        s_inputs = boltz_model.input_embedder(feats_detached)
    
    
    
    
    s_inputs = s_inputs.detach()
    
    print(f"\n:")
    print(f" featsinallalreadydetach(receptorMSA)")
    print(f" s_inputsalreadydetach(receptorkeepdetached,binderinneedgradient)")
    
    # ============================================================
    
    # ============================================================
    
    
    
    
    initial_avg_prob = None  
    if use_distogram_penalty and distogram_penalty_regions is not None and distogram_penalty_steps > 0:
        print(f"\n" + "="*80)
        print(f"1:Distogram Penaltyoptimization")
        print(f"="*80)
        print(f":region(framework-receptor pair)")
        print(f" step:{distogram_penalty_steps}")
        print(f" weight:{distogram_penalty_weight}")
        print(f" regionvs:{distogram_penalty_regions}")
        print(f" gradient:distogram_loss -> distogram -> distogram_module (Linear) -> z -> z_params")
        print(f" notneedPairformer,gradient")
        
        from dream_boltz2.losses.interface_losses import DistogramPenaltyLoss
        
        
        metadata_for_penalty = metadata.copy() if metadata else {}
        if optimizable_regions is not None:
            metadata_for_penalty['optimizable_regions'] = optimizable_regions
        
        distogram_penalty_loss_fn = DistogramPenaltyLoss(
            boltz_model=boltz_model,
            penalty_regions=distogram_penalty_regions,
            weight=distogram_penalty_weight,
        )
        
        
        distogram_penalty_lr = 0.1  
        distogram_penalty_optimizer = optim.Adam(params, lr=distogram_penalty_lr)
        print(f": {distogram_penalty_lr} (optimization,optimization)")
        
        
        initial_avg_prob = None  
        with torch.no_grad():
            s_init_check, z_init_check = assemble_full_sz()
            _, initial_info = distogram_penalty_loss_fn(
                z=z_init_check,
                metadata=metadata_for_penalty,
            )
        initial_avg_prob = initial_info.get('avg_penalty_prob', 1.0)
        print(f"\n:")
        print(f" contact: {initial_avg_prob:.4f}")
        print(f" contact: {initial_info.get('max_penalty_prob', 0):.4f}")
        print(f" contact: {initial_info.get('min_penalty_prob', 0):.4f}")
        print(f" incontact: {initial_info.get('median_penalty_prob', 0):.4f}")
        print(f" contact: (>40%): {initial_info.get('high_contact_count', 0)}, "
              f"in(30-40%): {initial_info.get('medium_contact_count', 0)}, "
              f"(≤30%): {initial_info.get('low_contact_count', 0)}")
        
        
        optimization_history = []
        
        print(f"\n optimization(10stepinfo)...")
        
        for step in range(distogram_penalty_steps):
            distogram_penalty_optimizer.zero_grad()
            
            
            s_optim, z_optim = assemble_full_sz()
            
            
            distogram_penalty_loss, distogram_penalty_info = distogram_penalty_loss_fn(
                z=z_optim,
                metadata=metadata_for_penalty,
            )
            
            
            avg_penalty_prob = distogram_penalty_info.get('avg_penalty_prob', 1.0)
            high_contact_count = distogram_penalty_info.get('high_contact_count', 0)
            loss_raw = distogram_penalty_info.get('loss_raw', 0)
            
            
            optimization_history.append({
                'step': step + 1,
                'avg_prob': avg_penalty_prob,
                'loss': distogram_penalty_loss.item(),
                'loss_raw': loss_raw,
                'max_prob': distogram_penalty_info.get('max_penalty_prob', 0),
                'min_prob': distogram_penalty_info.get('min_penalty_prob', 0),
                'high_contact': high_contact_count,
                'medium_contact': distogram_penalty_info.get('medium_contact_count', 0),
                'low_contact': distogram_penalty_info.get('low_contact_count', 0),
            })
            
            
            if high_contact_count == 0:
                print(f"\n contactvsalready0,")
                break
            elif high_contact_count < 5 and loss_raw < 0.01:
                print(f"\n contactvsalready{high_contact_count} (<5)lossalready({loss_raw:.6f}),")
                break
            
            
            distogram_penalty_loss.backward()
            
            
            z_grad_norm = 0.0
            s_grad_norm = 0.0
            total_grad_norm = 0.0
            num_params_with_grad = 0
            
            
            
            z_params_flat_grad_norm = 0.0
            if z_params_flat.grad is not None:
                z_params_flat_grad_norm = z_params_flat.grad.norm().item()
            
            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            param_grad_norm = param.grad.norm().item()
                            total_grad_norm += param_grad_norm ** 2
                            num_params_with_grad += 1
                            
                            if len(param.shape) == 3:  # s_params: [1, L_opt, D_s]
                                s_grad_norm = max(s_grad_norm, param_grad_norm)
                            elif len(param.shape) == 1:  # z_params_flat: [num_pairs * D_z]
                                z_grad_norm = max(z_grad_norm, param_grad_norm)
            
            total_grad_norm = total_grad_norm ** 0.5  # L2 norm
            
            
            if step == 0 or (step + 1) % 10 == 0 or high_contact_count == 0:
                if z_params_flat_grad_norm > 0 and z_grad_norm == 0:
                    print(f" warning: z_params_flatgradient({z_params_flat_grad_norm:.8f})z_grad_normas0,gradient！")
            
            
            distogram_penalty_optimizer.step()
            
            
            if step == 0 or (step + 1) % 10 == 0 or step == distogram_penalty_steps - 1 or high_contact_count == 0:
                change_from_initial = initial_avg_prob - avg_penalty_prob
                change_percent = (change_from_initial / initial_avg_prob * 100) if initial_avg_prob > 0 else 0.0
                
                
                loss_change = ""
                max_prob_change = ""
                if len(optimization_history) > 1:
                    prev_loss = optimization_history[-2]['loss']
                    curr_loss = optimization_history[-1]['loss']
                    loss_change = f"Loss: {curr_loss - prev_loss:+.6f}"
                    
                    prev_max_prob = optimization_history[-2]['max_prob']
                    curr_max_prob = distogram_penalty_info.get('max_penalty_prob', 0)
                    max_prob_change = f"contact: {curr_max_prob - prev_max_prob:+.6f}"
                
                print(f"\n  [Distogram Penalty Step {step+1}/{distogram_penalty_steps}]")
                print(f"    Loss: {distogram_penalty_loss.item():.6f} (raw: {distogram_penalty_info.get('loss_raw', 0):.6f}) {loss_change}")
                print(f" contact: {avg_penalty_prob:.6f} (: {initial_avg_prob:.6f},: {change_from_initial:+.6f} ({change_percent:+.2f}%))")
                print(f" contact: Max={distogram_penalty_info.get('max_penalty_prob', 0):.6f} {max_prob_change}, "
                      f"Min={distogram_penalty_info.get('min_penalty_prob', 0):.6f}, "
                      f"Median={distogram_penalty_info.get('median_penalty_prob', 0):.6f}, "
                      f"Std={distogram_penalty_info.get('std_penalty_prob', 0):.6f}")
                print(f" contact: (>40%): {distogram_penalty_info.get('high_contact_count', 0)}, "
                      f"in(30-40%): {distogram_penalty_info.get('medium_contact_count', 0)}, "
                      f"(≤30%): {distogram_penalty_info.get('low_contact_count', 0)}")
                print(f" gradient: s_grad={s_grad_norm:.8f}, z_grad={z_grad_norm:.8f}, "
                      f"total_grad={total_grad_norm:.8f}, gradientargument={num_params_with_grad}")
                if step == 0 or (step + 1) % 10 == 0 or high_contact_count == 0:
                    print(f" gradient: z_params_flat_grad={z_params_flat_grad_norm:.8f}")
                    if z_params_flat_grad_norm > 0 and z_grad_norm == 0:
                        print(f" warning: z_params_flatgradient({z_params_flat_grad_norm:.8f})z_grad_normas0,gradient！")
                
                
                if z_grad_norm < 1e-6:
                    print(f" warning: zgradient (<1e-6),optimizationno！")
                    print(f": 1) contactvszargumentnotinoptimizationargumentin 2) gradient 3) loss")
                if abs(change_from_initial) < 0.001:
                    print(f" warning: contactno (<0.001),optimizationno！")
                if change_from_initial < 0:
                    print(f" warning: contact,optimizationerror！")
                
                
                if step > 0 and high_contact_count == 0 and optimization_history[-2]['high_contact'] > 0:
                    prev_high_contact = optimization_history[-2]['high_contact']
                    print(f" note: contactvsfrom{prev_high_contact}vsstepto0vs,oroptimization！")
                    print(f": ifthisas,can;")
        
        
        final_avg_prob = distogram_penalty_info.get('avg_penalty_prob', 0)
        total_change = initial_avg_prob - final_avg_prob
        total_change_percent = (total_change / initial_avg_prob * 100) if initial_avg_prob > 0 else 0.0
        
        initial_high_contact = initial_info.get('high_contact_count', 0)
        final_high_contact = distogram_penalty_info.get('high_contact_count', 0)
        high_contact_change = initial_high_contact - final_high_contact
        
        print(f"\n Distogram Penaltyoptimization！")
        print(f" optimization:")
        print(f" contact: {initial_avg_prob:.4f}")
        print(f" contact: {final_avg_prob:.4f}")
        print(f": {total_change:+.4f} ({total_change_percent:+.1f}%)")
        print(f" optimizationstep: {len(optimization_history)}/{distogram_penalty_steps}")
        print(f"\n contactvsoptimization():")
        print(f" contactvs(>40%): {initial_high_contact} vs")
        print(f" contactvs(>40%): {final_high_contact} vs")
        print(f" contactvs: {high_contact_change:+d} vs ({high_contact_change/initial_high_contact*100 if initial_high_contact > 0 else 0:+.1f}%)")
        
        print(f"\n contact:")
        print(f" contact: {distogram_penalty_info.get('max_penalty_prob', 0):.4f}")
        print(f" contact: {distogram_penalty_info.get('min_penalty_prob', 0):.4f}")
        print(f" incontact: {distogram_penalty_info.get('median_penalty_prob', 0):.4f}")
        print(f": {distogram_penalty_info.get('std_penalty_prob', 0):.4f}")
        print(f" contact(>40%): {final_high_contact} vs")
        print(f" incontact(30-40%): {distogram_penalty_info.get('medium_contact_count', 0)} vs")
        print(f" contact(≤30%): {distogram_penalty_info.get('low_contact_count', 0)} vs")
        print(f" residuevs: {distogram_penalty_info.get('num_penalty_pairs', 0)}")
        
        
        if len(optimization_history) > 1:
            first_prob = optimization_history[0]['avg_prob']
            last_prob = optimization_history[-1]['avg_prob']
            trend = '' if last_prob < first_prob else '' if last_prob > first_prob else ''
            print(f"\n optimization: {trend} (from {first_prob:.4f} to {last_prob:.4f})")
            
            
            if abs(total_change) < 0.01:
                print(f" warning: contact (<0.01),optimizationno！")
                print(f": check, weightorgradientwhether")
        
        print(f" 1,2(diffusionoptimization)")
        
        
        
        s_init_after_penalty, z_init_after_penalty = assemble_full_sz()
        s_init_after_penalty = s_init_after_penalty.detach().clone()
        z_init_after_penalty = z_init_after_penalty.detach().clone()
        print(f" alreadysave1(s,z)(distogram),forPDBgeneratevs")
    else:
        print(f"\n[INFO] Distogram Penaltyoptimizationnot(distogram_penalty_steps=0)")
        print(f" 2(diffusionoptimization)")
        s_init_after_penalty = None
        z_init_after_penalty = None
    
    # ============================================================
    
    # ============================================================
    print(f"\n" + "="*80)
    print(f"2:diffusionoptimization(optimization)")
    print(f"="*80)
    print(f" step:{num_steps}")
    print(f":{lr}")
    print(f" threshold:loss < 0.5")
    print(f" uselossgenerate(RG loss, Helix loss, Hotspot loss)")
    print(f":notcontainsDistogram Penalty,1")
    
    
    
    
    
    history = {
        'loss': [],
        's_mean': [],
        's_std': [],
        'z_mean': [],
        'z_std': [],
    }
    
    
    early_stop_threshold = 0.5
    early_stopped = False
    
    
    if num_steps == 0:
        print(f"\n[INFO] num_steps=0,skipGSDoptimization,useZscaling(s,z)sampling")
        
        s_optim, z_optim = assemble_full_sz()
        print(f" already(s,z)(alreadyalphascaling)")
        print(f"  s shape: {s_optim.shape}")
        print(f"  z shape: {z_optim.shape}")
    else:
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # ============================================================
            
            # ============================================================
            
            s_optim, z_optim = assemble_full_sz()
            
            
            
            
            
            # ============================================================
            
            # ============================================================
        
        
        # 
        
        
        
        # 
        
        
        
        
        
        
        # 
            
            
            coords = diffusion.single_step_sds(
                s_trunk=s_optim,  
                z_trunk=z_optim,  
                feats=feats,
                sigma=250.0,  
                s_inputs=s_inputs,
                center_coords=True,
                
                
                
                
            )
            
            # ============================================================
            
            # ============================================================
            
            loss, loss_info = compute_structure_loss(
                pred_coords=coords,
                target_coords=target_coords,
                feats=feats,
                metadata=metadata,
                
                use_rg_loss=use_rg_loss,
                use_helix_loss=use_helix_loss,
                use_hotspot_loss=use_hotspot_loss,
                
                rg_weight=rg_weight,
                helix_weight=helix_weight,
                hotspot_weight=hotspot_weight,
                
                hotspot_indices=hotspot_indices,
            )
            
            loss_value = loss.item()
            
            
            if loss_value < early_stop_threshold:
                print(f"\n:loss ({loss_value:.4f}) < {early_stop_threshold},optimization")
                early_stopped = True
                
                
            
            # ============================================================
            
            # ============================================================
            
            
            
            
            
            # ============================================================
            
            # ============================================================
            
            # 
            
            
            
            loss.backward()
            
            # ============================================================
            
            # ============================================================
            if step == 0 or (step + 1) % 50 == 0:
                
                if s_params.grad is not None:
                    s_grad_norm = s_params.grad.norm().item()
                    print(f" [Gradient Check] s_paramsgradientnorm={s_grad_norm:.6f} (optimizationregion)")
                if z_params_flat.grad is not None:
                    z_grad_norm = z_params_flat.grad.norm().item()
                    print(f" [Gradient Check] z_params_flatgradientnorm={z_grad_norm:.6f} (optimizationvs)")
            
            
            s_grad_norm = s_params.grad.norm().item() if s_params.grad is not None else 0.0
            z_grad_norm = z_params_flat.grad.norm().item() if z_params_flat.grad is not None else 0.0
            
            # ============================================================
            
            # ============================================================
            optimizer.step()
            
            # ============================================================
            
            # ============================================================
            history['loss'].append(loss.item())
            
            
            if early_stopped:
                break
            history['s_mean'].append(s_optim.mean().item())
            history['s_std'].append(s_optim.std().item())
            history['z_mean'].append(z_optim.mean().item())
            history['z_std'].append(z_optim.std().item())
            
            
            if 'rg_loss' in loss_info:
                if 'rg_value' in loss_info['rg_loss']:
                    if 'rg_value' not in history:
                        history['rg_value'] = []
                    history['rg_value'].append(loss_info['rg_loss']['rg_value'])
            
            if 'helix_loss' in loss_info:
                if 'raw_loss' in loss_info['helix_loss']:
                    if 'helix_loss' not in history:
                        history['helix_loss'] = []
                    history['helix_loss'].append(loss_info['helix_loss']['raw_loss'])
            
            if 'hotspot_loss' in loss_info:
                if 'raw_loss' in loss_info['hotspot_loss']:
                    if 'hotspot_loss' not in history:
                        history['hotspot_loss'] = []
                    history['hotspot_loss'].append(loss_info['hotspot_loss']['raw_loss'])
            
            # ============================================================
            
            # ============================================================
            if step % 10 == 0 or step == num_steps - 1:
                
                loss_str = f"Loss={loss.item():.4f}"
                if 'rg_loss' in loss_info and 'rg_value' in loss_info['rg_loss']:
                    loss_str += f" | RG={loss_info['rg_loss']['rg_value']:.2f}"
                if 'helix_loss' in loss_info and 'raw_loss' in loss_info['helix_loss']:
                    loss_str += f" | Helix={loss_info['helix_loss']['raw_loss']:.4f}"
                if 'hotspot_loss' in loss_info and 'raw_loss' in loss_info['hotspot_loss']:
                    loss_str += f" | Hotspot={loss_info['hotspot_loss']['raw_loss']:.4f}"
                
                
                mem_str = ""
                if monitor_memory:
                    current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                    mem_str = f" | Mem: {current_memory:.2f}GB (Peak: {peak_memory:.2f}GB)"
                
                print(
                    f"[Step {step+1:3d}/{num_steps}] "
                    f"{loss_str} | "
                    f"s_grad={s_grad_norm:.4f} | "
                    f"z_grad={z_grad_norm:.4f}{mem_str}"
                )
    
    print("\n" + "="*80)
    if num_steps == 0:
        print('skipoptimization(num_steps=0)')
    else:
        print('optimization！')
    print("="*80)
    if len(history['loss']) > 0:
        print(f"loss: {history['loss'][-1]:.4f}")
        print(f"loss: {history['loss'][0]:.4f}")
        print(f"Loss: {history['loss'][0] - history['loss'][-1]:.4f}")
    else:
        print(f"notoptimization(num_steps=0),use(s,z)sampling")
    
    
    if use_distogram_penalty and distogram_penalty_regions is not None:
        print(f"\n" + "="*80)
        print(f"2:Distogram(forvs)")
        print(f"="*80)
        with torch.no_grad():
            s_final, z_final = assemble_full_sz()
            _, final_distogram_info = distogram_penalty_loss_fn(
                z=z_final,
                metadata=metadata_for_penalty,
            )
        
        final_avg_prob = final_distogram_info.get('avg_penalty_prob', 0)
        print(f" (2):")
        print(f" contact: {final_avg_prob:.4f}")
        print(f" contact: {final_distogram_info.get('max_penalty_prob', 0):.4f}")
        print(f" contact: {final_distogram_info.get('min_penalty_prob', 0):.4f}")
        print(f" incontact: {final_distogram_info.get('median_penalty_prob', 0):.4f}")
        print(f": {final_distogram_info.get('std_penalty_prob', 0):.4f}")
        print(f" contact:")
        print(f" contact(>40%): {final_distogram_info.get('high_contact_count', 0)} vs")
        print(f" incontact(30-40%): {final_distogram_info.get('medium_contact_count', 0)} vs")
        print(f" contact(≤30%): {final_distogram_info.get('low_contact_count', 0)} vs")
        print(f" residuevs: {final_distogram_info.get('num_penalty_pairs', 0)}")
        
        
        if 'initial_avg_prob' in locals():
            change_from_initial = initial_avg_prob - final_avg_prob
            change_percent = (change_from_initial / initial_avg_prob * 100) if initial_avg_prob > 0 else 0.0
            print(f"\n vs(1 vs 2):")
            print(f" contact: {initial_avg_prob:.4f}")
            print(f" contact: {final_avg_prob:.4f}")
            print(f": {change_from_initial:+.4f} ({change_percent:+.1f}%)")
            
            
            if abs(change_from_initial) < 0.01:
                print(f" warning: contactno,distogram penaltyoptimizationno！")
                print(f":")
                print(f" 1. ,nooptimization")
                print(f" 2. gradient(distogramz,gradient)")
                print(f" 3. contactalready,optimizationempty")
                print(f" 4. optimizationstepnot")
            elif change_from_initial > 0:
                print(f" contact,optimization")
            else:
                print(f" contact,exists")
    
    
    if monitor_memory:
        final_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        reserved_memory = torch.cuda.memory_reserved() / (1024**3)  # GB
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        print(f"\nuse:")
        print(f": {initial_memory:.2f} GB")
        print(f": {final_memory:.2f} GB")
        print(f": {peak_memory:.2f} GB")
        print(f": {reserved_memory:.2f} GB")
        print(f" times: {peak_memory - initial_memory:.2f} GB")
        
        
        
        
        
        # 
        
        
        
        
        
        model_memory = initial_memory  
        single_design_activation = peak_memory - initial_memory  
        single_design_params = 0.02  
        
        
        available_memory = total_gpu_memory - peak_memory
        
        batch_10_extra = 10 * (single_design_activation + single_design_params) - (single_design_activation + single_design_params)
        
        if single_design_activation > 0:
            
            estimated_batch_size = int(available_memory / (single_design_activation + single_design_params))
            
            print(f"\nBatch:")
            print(f" GPU: {total_gpu_memory:.2f} GB")
            print(f" current: {peak_memory:.2f} GB")
            print(f": {available_memory:.2f} GB")
            print(f" times: {single_design_activation:.2f} GB")
            print(f" timesargument: {single_design_params:.2f} GB")
            print(f" batch=10(): {batch_10_extra:.2f} GB")
            print(f" supportbatch: ~{estimated_batch_size} ()")
            
            if available_memory >= batch_10_extra:
                print(f" cansupportbatch=10(weight)")
                print(f" batch=10: {peak_memory + batch_10_extra:.2f} GB")
            elif estimated_batch_size >= 5:
                print(f" cansupportbatch={estimated_batch_size},batch=5")
            else:
                print(f" nosupportbatch=10,batch={max(1, estimated_batch_size)}")
    
    # ============================================================
    
    # ============================================================
    
    
    # print(f"\n" + "="*80)
    
    
    
    return s_optim, z_optim, history, s_init, z_init, s_init_after_penalty, z_init_after_penalty


def _build_optimize_arg_parser():
    """ build optimize arg parser."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Dream backbones by optimizing Boltz-2 single and pair representations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m dream_boltz2.cli.design --config examples/trop2_fluorosulfate.yaml "
            "--checkpoint /path/to/boltz2.ckpt\n"
        )
    )
    
    
    parser.add_argument('--config', type=str, default=None,
                        help=' YAML configfile path( Boltz2 ).'
                             'useconfigfile,argumentconfigfilein.')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Boltz2 checkpoint path')
    parser.add_argument('--pdb', type=str, default=None, help='PDB file path (optional, for target coords)')
    parser.add_argument('--receptor_chain', type=str, default='B', help='Receptor chain ID')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_steps', type=int, default=200, help='Number of optimization steps (default: 200)')
    parser.add_argument('--lr', type=float, default=0.2, help='Learning rate (default: 0.2)')
    
    parser.add_argument('--early_filter_threshold', type=float, default=1.0, help='Early skeleton filter threshold: if final loss after optimization >= this value, regenerate skeleton (default: 1.0)')
    parser.add_argument('--max_regeneration_attempts', type=int, default=10, help='Maximum number of regeneration attempts when early filter fails (default: 10)')
    
    parser.add_argument('--receptor_msa', type=str, default=None, help='Receptor MSA file path (.a3m format)')
    parser.add_argument('--max_msa_seqs', type=int, default=700, help='Maximum MSA sequences (default: 700)')
    
    parser.add_argument('--use_rg_loss', action='store_true', default=True, help='Use RG loss')
    parser.add_argument('--use_helix_loss', action='store_true', default=True, help='Use Helix loss')
    parser.add_argument('--use_hotspot_loss', action='store_true', default=True, help='Use Hotspot loss')
    parser.add_argument('--rg_weight', type=float, default=1.0, help='RG loss weight')
    parser.add_argument('--helix_weight', type=float, default=0.5, help='Helix loss weight')
    parser.add_argument('--hotspot_weight', type=float, default=0.3, help='Hotspot loss weight')
    parser.add_argument('--hotspot_indices', type=str, default=None, help='Hotspot indices (comma-separated, 1-based, e.g., "234,240,250")')
    
    parser.add_argument('--use_distogram_penalty', action='store_true', help='Use distogram penalty loss to weaken binding probability of specified regions')
    parser.add_argument('--distogram_penalty_weight', type=float, default=1.0, help='Distogram penalty loss weight')
    parser.add_argument('--distogram_penalty_regions', type=str, default=None, help='Regions to penalize (format: "region1:region2,region3:region4"). Region format: "start:end" (0-based, e.g., "0:50") or "binder_framework" or "binder" (all binder regions) or "receptor". Default: "binder:receptor" (includes all binder regions, CDR + framework). Example: "binder:receptor" or "binder_framework:receptor" or "0:50:100:150"')
    parser.add_argument('--distogram_penalty_steps', type=int, default=0, help='Number of steps for distogram penalty pre-optimization (0 means disabled, >0 means run distogram penalty optimization before structure optimization)')
    
    parser.add_argument('--contact_constraints', type=str, default=None, help='Contact constraints: "binder_idx1:receptor_idx1:distance1,binder_idx2:receptor_idx2:distance2" (e.g., "5:40:8.0,10:55:8.0"). allindexas1-based')
    
    parser.add_argument('--binder_template', type=str, default=None, help='Binder initial sequence template (e.g., "AAA", "WWW", "XXX"). If None, uses default SUPER_RESIDUE (X, unknown residue)')
    
    parser.add_argument('--nanobody', action='store_true', help='Enable nanobody mode: only optimize CDR regions, fix framework regions')
    parser.add_argument('--fab', action='store_true', help='Enable Fab mode: only optimize CDR regions (6 CDRs: 3 on H chain + 3 on L chain), fix framework regions')
    parser.add_argument('--cdr_regions', type=str, default=None, help='CDR regions to optimize (format: "start1:end1,start2:end2,start3:end3" or "nanobody"/"fab" for default CDR regions). Example: "26:34,52:59,98:118"')
    parser.add_argument('--nanobody_template', type=str, default='nanobody_default', help='Nanobody template name. Options: nanobody_default (fixed 8coh, backward compat), random (4-template system), 7eow, 7xl0, 8coh, 8z8v')
    parser.add_argument('--fab_template', type=str, default='fab_default', help='Fab template name. Options: fab_default (typical Fab with H chain ~220-230 residues + L chain ~210-220 residues)')
    
    parser.add_argument('--cyclic', action='store_true', help='Enable cyclic peptide mode for binder. The binder will be treated as a cyclic peptide (N-terminus connected to C-terminus)')
    
    parser.add_argument('--ligand', type=str, default=None, help='Ligand specification. Format: "id:ccd" (e.g., "C:GDP") or "id:smiles:SMILES_STRING" (e.g., "C:smiles:CC(=O)O"). Multiple ligands: "C:GDP,D:ATP"')
    
    
    parser.add_argument('--template_file', type=str, default=None, help='Template PDB/CIF file path (forconstraint,GSD)')
    parser.add_argument('--template_chain_id', type=str, default=None, help='Template chain ID (e.g., "B" for receptor chain)')
    parser.add_argument('--template_force', action='store_true', default=True, help='Use template force constraint (templateconstraint,defaultTrue,and)')
    parser.add_argument('--no_template_force', dest='template_force', action='store_false', help='template forceconstraint')
    parser.add_argument('--template_threshold', type=float, default=3.0, help='Template force threshold (Å, default3.0,and)')
    
    parser.add_argument('--anti_contact_residues', type=str, default=None, help='Receptorresidueregion,mutationbinder(:"100-120,200-210",1-based,)')
    parser.add_argument('--anti_contact_mutation_type', type=str, default='N', choices=['N', 'Q', 'G', 'A', 'P'], help='mutation(defaultN,,)')
    
    parser.add_argument('--required_contacts', type=str, default=None, help='Receptorresiduepositionlist,autofromCDRbinderresiduevs(:"70,55,97",1-based,)')
    
    parser.add_argument('--fixed_residue_contacts', type=str, default=None, help='fixedresiduevs,:"93H-76R,45E-120K"(1-based)or"CDR3_H-76R"(inCDR3position).support:1)position:93H-76R 2)CDR:CDR3_H-76R.thisresidueinsequenceoptimizationinnotMPNN')
    
    parser.add_argument('--mode', type=int, default=1, choices=[1, 2], help='mode: 1=timesoptimization(default), 2=mode(outputPDB)')
    parser.add_argument('--num_iterations', type=int, default=5, help='mode2times(default5times)')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory(if,allPDBfilesavetodirectory;ifnot,savetocurrentdirectory)')
    
    
    parser.add_argument('--creativity', type=float, default=0.0,
                        help=' alphafactor [0, 1].0=mode(default),1=alpha.'
                             'in Pairformer round recycling scaling Z_ij(pairwise)residue,'
                             'diffusionto"generate".'
                             ':scaling receptor-receptor all Z_ij;'
                             'Nanobody/Fab:scaling CDR Z_ij.'
                             ':0.3-0.7.')
    
    
    parser.add_argument('--num_design_rounds', type=int, default=1,
                        help=' GSD optimizationround(roundusenot).default 1.'
                             'roundcangenerate.')
    parser.add_argument('--samples_per_round', type=int, default=1,
                        help=' round GSD optimizationsampling.default 1.'
                             'eachusenot,diffusiongenerate.'
                             ' = num_design_rounds × samples_per_round.')
    
    return parser


def _emit_mode1_round_diffusion_and_pdbs(
    *,
    diffusion,
    args,
    config,
    nanobody_mode: bool,
    fab_mode: bool,
    fab_chain_info,
    round_idx_1based: int,
    num_design_rounds: int,
    samples_per_round: int,
    s_init,
    z_init,
    s_optim,
    z_optim,
    s_init_after_penalty,
    z_init_after_penalty,
    round_feats: Dict,
    round_metadata: Dict,
    cdr_regions_binder_local: Optional[List[Tuple[int, int]]],
    cdr_regions_fallback: Optional[List[Tuple[int, int]]],
    generated_pdbs_accumulator: List[str],
) -> None:
    """ emit mode1 round diffusion and pdbs."""
    if write_coords_to_pdb is None:
        return

    from pathlib import Path

    print("\n" + "=" * 80)
    print(
        f" round({round_idx_1based}/{num_design_rounds})"
        f":diffusionsampling PDB(andround (s,z), feats vs)"
    )
    print("=" * 80)

    output_dir: Optional[Path] = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n output directory: {output_dir.absolute()}")
    else:
        print(f"\n output directory: currentdirectory(not--output_dir)")

    _feats = round_feats
    _meta = round_metadata
    if cdr_regions_binder_local:
        _cdr_for_pdb = cdr_regions_binder_local
    else:
        _cdr_for_pdb = cdr_regions_for_pdb_from_metadata(_meta, cdr_regions_fallback)

    feats_for_pdb = {}
    for k, v in _feats.items():
        if isinstance(v, torch.Tensor):
            feats_for_pdb[k] = v.cpu()
        else:
            feats_for_pdb[k] = v

    feats_for_diffusion = {}
    for k, v in _feats.items():
        if isinstance(v, torch.Tensor):
            feats_for_diffusion[k] = v.to(args.device)
        else:
            feats_for_diffusion[k] = v

    rp = "" if num_design_rounds <= 1 else f"round{round_idx_1based:02d}_"

    metadata_for_pdb = _meta.copy()
    if config.receptor_chains and len(config.receptor_chains) > 0:
        metadata_for_pdb['receptor_chain_id'] = config.receptor_chains[0]
    if config.binder_chains and len(config.binder_chains) > 0:
        metadata_for_pdb['binder_chain_id'] = config.binder_chains[0]
    if fab_mode and fab_chain_info:
        binder_slice = _meta['binder_slice']
        h_chain_slice = slice(binder_slice.start, binder_slice.start + fab_chain_info['H_chain_length'])
        l_chain_slice = slice(binder_slice.start + fab_chain_info['H_chain_length'], binder_slice.stop)
        metadata_for_pdb['binder_chains'] = [('C', h_chain_slice), ('D', l_chain_slice)]

    _meta_struct = _meta

    
    print('\n[1/2] useround(s,z)diffusionsampling…')
    print(f" res_type shape: {_feats['res_type'].shape if 'res_type' in _feats else 'N/A'}")

    try:
        with torch.no_grad():
            coords_init = diffusion.forward(
                s=s_init.to(args.device),
                z=z_init.to(args.device),
                feats=feats_for_diffusion,
                num_steps=25,
                use_fixed_noise=True,
                start_coords=None,
                use_physical_guidance=False,
            )
        coords_init_cpu = coords_init.cpu()
        if 'token_to_center_atom' in feats_for_pdb:
            ttc = feats_for_pdb['token_to_center_atom']
            if ttc.dim() == 2:
                ttc = ttc.unsqueeze(0)
            all_ca_init = torch.bmm(ttc.float(), coords_init_cpu)[0]
        elif 'token_to_rep_atom' in feats_for_pdb:
            ttr = feats_for_pdb['token_to_rep_atom']
            if ttr.dim() == 2:
                ttr = ttr.unsqueeze(0)
            all_ca_init = torch.bmm(ttr.float(), coords_init_cpu)[0]
        else:
            raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")

        if s_init_after_penalty is not None and z_init_after_penalty is not None:
            with torch.no_grad():
                coords_ap = diffusion.forward(
                    s=s_init_after_penalty.to(args.device),
                    z=z_init_after_penalty.to(args.device),
                    feats=feats_for_diffusion,
                    num_steps=25,
                    use_fixed_noise=True,
                    start_coords=None,
                    use_physical_guidance=False,
                )
            coords_ap_cpu = coords_ap.cpu()
            if 'token_to_center_atom' in feats_for_pdb:
                ttc = feats_for_pdb['token_to_center_atom']
                if ttc.dim() == 2:
                    ttc = ttc.unsqueeze(0)
                ca_ap = torch.bmm(ttc.float(), coords_ap_cpu)[0]
            else:
                ttr = feats_for_pdb['token_to_rep_atom']
                if ttr.dim() == 2:
                    ttr = ttr.unsqueeze(0)
                ca_ap = torch.bmm(ttr.float(), coords_ap_cpu)[0]
            fn_ap = (output_dir / f"{rp}structure_initial_sz_after_penalty.pdb") if output_dir else Path(f"{rp}structure_initial_sz_after_penalty.pdb")
            meta_ap = _meta_struct.copy()
            if config.receptor_chains:
                meta_ap['receptor_chain_id'] = config.receptor_chains[0]
            if config.binder_chains:
                meta_ap['binder_chain_id'] = config.binder_chains[0]
            if fab_mode and fab_chain_info:
                bs = _meta_struct['binder_slice']
                meta_ap['binder_chains'] = [
                    ('C', slice(bs.start, bs.start + fab_chain_info['H_chain_length'])),
                    ('D', slice(bs.start + fab_chain_info['H_chain_length'], bs.stop)),
                ]
            write_coords_to_pdb(
                coords=coords_ap_cpu,
                feats=feats_for_pdb,
                metadata=meta_ap,
                output_file=str(fn_ap),
                ca_coords=ca_ap,
                cdr_regions=_cdr_for_pdb if nanobody_mode else None,
            )
            generated_pdbs_accumulator.append(str(fn_ap))
            print(f" {fn_ap}")

        fn_init = (output_dir / f"{rp}structure_initial_sz_real_sequence.pdb") if output_dir else Path(f"{rp}structure_initial_sz_real_sequence.pdb")
        meta_init = metadata_for_pdb.copy()
        write_coords_to_pdb(
            coords=coords_init_cpu,
            feats=feats_for_pdb,
            metadata=meta_init,
            output_file=str(fn_init),
            ca_coords=all_ca_init,
            use_backbone=False,
            use_full_sidechain=True,
            cdr_regions=_cdr_for_pdb if nanobody_mode else None,
        )
        generated_pdbs_accumulator.append(str(fn_init))
        print(f" {fn_init}")
    except Exception as e:
        print(f" [1/2] round PDB failed: {e}")
        import traceback
        traceback.print_exc()

    
    print(f"\n[2/2] roundoptimization(s,z) diffusionsampling × {samples_per_round} …")

    if 'token_bonds' in feats_for_diffusion:
        tbf = feats_for_diffusion['token_bonds']
        if isinstance(tbf, torch.Tensor) and tbf.numel() > 0:
            bc = (tbf > 0).sum().item()
            print(f" token_bonds: {bc}")

    for sample_idx in range(samples_per_round):
        try:
            with torch.no_grad():
                coords_o = diffusion.forward(
                    s=s_optim.to(args.device),
                    z=z_optim.to(args.device),
                    feats=feats_for_diffusion,
                    num_steps=25,
                    use_fixed_noise=False,
                    start_coords=None,
                    use_physical_guidance=False,
                )
            coords_o_cpu = coords_o.cpu()
            if 'token_to_center_atom' in feats_for_pdb:
                ttc = feats_for_pdb['token_to_center_atom']
                if ttc.dim() == 2:
                    ttc = ttc.unsqueeze(0)
                all_ca_o = torch.bmm(ttc.float(), coords_o_cpu)[0]
            elif 'token_to_rep_atom' in feats_for_pdb:
                ttr = feats_for_pdb['token_to_rep_atom']
                if ttr.dim() == 2:
                    ttr = ttr.unsqueeze(0)
                all_ca_o = torch.bmm(ttr.float(), coords_o_cpu)[0]
            else:
                raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")

            if num_design_rounds > 1:
                if output_dir:
                    fn_o = output_dir / f"structure_round{round_idx_1based:02d}_sample{sample_idx + 1:02d}.pdb"
                else:
                    fn_o = Path(f"structure_round{round_idx_1based:02d}_sample{sample_idx + 1:02d}.pdb")
            elif samples_per_round == 1:
                fn_o = (output_dir / "structure_optimized_sz.pdb") if output_dir else Path("structure_optimized_sz.pdb")
            else:
                fn_o = (
                    output_dir / f"structure_sample{sample_idx + 1:02d}.pdb"
                    if output_dir
                    else Path(f"structure_sample{sample_idx + 1:02d}.pdb")
                )

            write_coords_to_pdb(
                coords=coords_o_cpu,
                feats=feats_for_pdb,
                metadata=metadata_for_pdb,
                output_file=str(fn_o),
                ca_coords=all_ca_o,
                use_backbone=False,
                use_full_sidechain=True,
                cdr_regions=_cdr_for_pdb if nanobody_mode else None,
            )
            generated_pdbs_accumulator.append(str(fn_o))
            print(f" {sample_idx + 1}/{samples_per_round}: {fn_o}")
        except Exception as e:
            print(f" {sample_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n round({round_idx_1based}/{num_design_rounds})PDB ")


def _run_with_args(args, parser=None):
    """ run with args."""
    from pathlib import Path  

    # =========================================================================
    
    # =========================================================================
    if args.config:
        try:
            from dream_boltz2.config import parse_dream_config, validate_config
        except ImportError:
            
            import sys
            script_dir = Path(__file__).parent.parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            from dream_boltz2.config import parse_dream_config, validate_config

        injected = getattr(args, '_dream_config', None) is not None

        if injected:
            print(' usealready Dream config(skip YAML ,ligand)')
            print("="*80)
            config = args._dream_config
            errors = validate_config(config)
            if errors:
                print(' configfilefailed:')
                for err in errors:
                    print(f"  - {err}")
                return

            print(f" configvsalready(): {args.config}")
        else:
            print(' load YAML configfile')
            print("="*80)
            config = parse_dream_config(args.config)
            errors = validate_config(config)

            if errors:
                print(' configfilefailed:')
                for err in errors:
                    print(f"  - {err}")
                return

            print(f" configfileloadok: {args.config}")
        print(f"  Receptor chains: {config.receptor_chains}")
        print(f"  Binder chains: {config.binder_chains}")
        print(f"  Design type: {config.design_type}")
        print(f"  Creativity: {config.creativity}")
        print(f"  Design rounds: {config.num_design_rounds}")
        print(f"  Samples per round: {config.samples_per_round}")
        
        if config.protected_regions.protected_modifications:
            print(f" Protected modifications: {config.protected_regions.protected_modifications}")
        if config.protected_regions.protected_ligands:
            print(f" Protected ligands: {config.protected_regions.protected_ligands}")
        if config.get_ligands():
            print(f" Ligands: {[lig.chain_id for lig in config.get_ligands()]}")
        
        
        yaml_ligands = config.get_ligands()
        if yaml_ligands:
            print(f" [DEBUG] YAMLinto {len(yaml_ligands)} ligand")
            for lig in yaml_ligands:
                print(f"    - chain_id={lig.chain_id}, ccd={lig.ccd}, smiles={lig.smiles}")
        
        if not args.ligand and yaml_ligands:
            ligand_specs = []
            for lig in yaml_ligands:
                if lig.ccd:
                    ligand_specs.append(f"{lig.chain_id}:{lig.ccd}")
                elif lig.smiles:
                    ligand_specs.append(f"{lig.chain_id}:smiles:{lig.smiles}")
            if ligand_specs:
                args.ligand = ','.join(ligand_specs)
                print(f" fromYAMLconfiginreadligands: {args.ligand}")
        elif args.ligand:
            print(f" [DEBUG] useargumentinligand: {args.ligand}")
        else:
            print(f" [DEBUG] nottoligandconfig(YAMLor)")
        
        
        yaml_constraints = None
        if hasattr(config, 'constraints') and config.constraints:
            yaml_constraints = config.constraints
            print(f" [CONSTRAINTS] fromYAMLconfiginread {len(yaml_constraints)} constraint")
            for constraint in yaml_constraints:
                constraint_type = constraint.constraint_type
                print(f"    - {constraint_type}: {constraint.data}")
        
        
        cli_args = config.to_cli_args()
        
        
        for key, value in cli_args.items():
            if value is not None and not hasattr(args, f'_{key}_set'):
                if parser is None:
                    setattr(args, key, value)
                elif getattr(args, key, None) == parser.get_default(key):
                    setattr(args, key, value)
        
        
        if config.design_type == 'nanobody' and not args.nanobody:
            args.nanobody = True
        elif config.design_type == 'fab' and not args.fab:
            args.fab = True
        
        
        if config.creativity > 0 and args.creativity == 0:
            args.creativity = config.creativity
        
        
        if config.num_design_rounds > 1 and args.num_design_rounds == 1:
            args.num_design_rounds = config.num_design_rounds
        if config.samples_per_round > 1 and args.samples_per_round == 1:
            args.samples_per_round = config.samples_per_round
        
        
        args._dream_config = config
        
        print()

        
        if getattr(config, 'ligand_scan', False):
            import sys
            _scripts_dir = Path(__file__).resolve().parent
            if str(_scripts_dir) not in sys.path:
                sys.path.insert(0, str(_scripts_dir))
            import ligand_scan as _ligand_scan
            if _ligand_scan.run_ligand_scan_if_requested(args):
                return
    
    print("="*80)
    print('DREAM-boltz2: optimization(s,z)generate')
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    if args.receptor_msa:
        print(f"Receptor MSA: {args.receptor_msa}")
        print(f"Max MSA seqs: {args.max_msa_seqs}")
    else:
        print("Receptor MSA: None (single sequence mode)")
    if args.binder_template:
        print(f"Binder Template: {args.binder_template}")
    else:
        print(f"Binder Template: None (using default SUPER_RESIDUE='X' - unknown residue for flexible initialization)")
    if args.cyclic:
        print(f" Cyclic Peptide Mode: Enabled (binder N-terminus will be connected to C-terminus)")
    
    
    nanobody_mode = args.nanobody
    fab_mode = args.fab
    cdr_regions = None
    nanobody_sequence = None
    fab_sequence = None
    fab_chain_info = None
    
    if nanobody_mode and fab_mode:
        raise ValueError("Cannot enable both --nanobody and --fab at the same time")
    
    if fab_mode:
        print(f"\n Fabmodealready")
        print(f"  Fab Template: {args.fab_template}")
        
        
        fab_sequence, default_cdr_regions, fab_chain_info = get_fab_template(args.fab_template)
        print(f" Fabsequencetemplatelength: {len(fab_sequence)} residue")
        print(f" Hchainlength: {fab_chain_info['H_chain_length']} residue")
        print(f" Lchainlength: {fab_chain_info['L_chain_length']} residue")
        print(f" defaultCDRregion (0-based, ,vssequence):")
        print(f" HchainCDR:")
        for i, (start, end) in enumerate(default_cdr_regions[:3], 1):
            print(f"      H-CDR{i}: [{start}, {end}) (1-based: {start+1}-{end}, {end-start}residue)")
        print(f" LchainCDR:")
        for i, (start, end) in enumerate(default_cdr_regions[3:], 1):
            print(f"      L-CDR{i}: [{start}, {end}) (1-based: {start+1}-{end}, {end-start}residue)")
        
        
        if args.cdr_regions:
            if args.cdr_regions.lower() == 'fab':
                cdr_regions = default_cdr_regions
                print(f"\n usedefaultFab CDRregion")
            else:
                
                cdr_regions = parse_cdr_regions(args.cdr_regions, len(fab_sequence), is_fab=True)
                print(f"\n useCDRregion")
        else:
            
            cdr_regions = default_cdr_regions
            print(f"\n usedefaultFab CDRregion(not--cdr_regions)")
        
        
        
        
        print(f"\n Fabmode:usechain(HchainandLchain)")
        print(f" Hchain: {fab_chain_info['H_chain_length']} residue")
        print(f" Lchain: {fab_chain_info['L_chain_length']} residue")
        print(f" frameworkusesequence,CDRuse'X'(optimization)")
    
    if nanobody_mode:
        print(f"\n Nanobodymodealready")
        print(f"  Nanobody Template: {args.nanobody_template}")

        
        _use_new_template = args.nanobody_template in ('random', '7eow', '7xl0', '8coh', '8z8v')
        if _use_new_template:
            nanobody_sequence, default_cdr_regions, _tmpl_used = generate_nanobody_sequence(args.nanobody_template)
            print(f" 4-template system: selected '{_tmpl_used}'")
            print(f" Nanobodysequencetemplatelength: {len(nanobody_sequence)} residue")
            print(f"  CDR filled with Y/F/R/D/W (weighted random)")
            for i, (start, end) in enumerate(default_cdr_regions, 1):
                print(f"    CDR{i}: [{start}, {end}) ({end-start}residue)")
        else:
            
            nanobody_sequence, default_cdr_regions = get_nanobody_template(args.nanobody_template)
            print(f" Nanobodysequencetemplatelength: {len(nanobody_sequence)} residue")
            print(f" defaultCDRregion (0-based, ):")
            for i, (start, end) in enumerate(default_cdr_regions, 1):
                print(f"    CDR{i}: [{start}, {end}) (1-based: {start+1}-{end}, {end-start}residue)")
        
        
        if args.cdr_regions:
            if args.cdr_regions.lower() == 'nanobody':
                cdr_regions = default_cdr_regions
                if _use_new_template:
                    print(f"\n usetemplate '{_tmpl_used}' CDRregion(length)")
                else:
                    print(f"\n usedefaultnanobody CDRregion(fixedlength)")
            else:
                
                
                cdr_regions = parse_cdr_regions(args.cdr_regions, binder_length=118)
                print(f"\n useCDRregion (0-based, ):")
                for i, (start, end) in enumerate(cdr_regions, 1):
                    print(f"    CDR{i}: [{start}, {end}) (1-based: {start+1}-{end}, {end-start}residue)")
        else:
            cdr_regions = default_cdr_regions
            if _use_new_template:
                print(f"\n usetemplate '{_tmpl_used}' CDRregion(length)")
            else:
                print(f"\n usedefaultnanobody CDRregion(fixedlength)")

        
        
        if args.binder_template and args.binder_template != nanobody_sequence:
            print(f"\n ℹ YAMLinbindersequence({len(args.binder_template)}residue)-> nanobodytemplate({len(nanobody_sequence)}residue)")
        args.binder_template = nanobody_sequence
        if _use_new_template:
            print(f" Bindersequence = {_tmpl_used} template({len(nanobody_sequence)}residue,CDR: Y/F/R/D/W)")
        else:
            print(f" Bindersequence = nanobody_default template({len(nanobody_sequence)}residue,CDR: X)")
    
    print("="*80)
    
    
    boltz_model = load_boltz_model(args.checkpoint, args.device)
    
    
    feature_prep = FeaturePreparatorComplex(
        boltz_model=boltz_model,
        device=args.device,
        mode='complex'
    )
    
    
    fixed_residue_contacts = None
    fixed_binder_positions = set()
    fixed_receptor_positions = set()
    fixed_binder_aa_map = {}
    fixed_contact_pairs = []
    
    
    
    
    
    
    
    fixed_residue_contacts_parsed = False
    if False and args.fixed_residue_contacts:  
        print(f"\n{'='*60}")
        print(f" parseFixed Residue Contacts(fixedresiduevs)")
        print(f"{'='*60}")
        
        fixed_residue_contacts, fixed_binder_positions, fixed_receptor_positions, fixed_binder_aa_map, fixed_contact_pairs = parse_fixed_residue_contacts(
            args.fixed_residue_contacts,
            receptor_seq,  
            cdr_regions if cdr_regions else [],
            binder_length,
            verbose=True
        )
        
        print(f"\n fixedresidue:")
        print(f" fixedvs: {len(fixed_residue_contacts)}")
        print(f" fixedbinderposition: {sorted([idx + 1 for idx in fixed_binder_positions])} (1-based)")
        print(f" fixedreceptorposition: {sorted([idx + 1 for idx in fixed_receptor_positions])} (1-based)")
        print(f" thispositioninsequenceoptimizationnotMPNN")
        print(f" thisvsinfake MSAin")
        print(f" ifthisreceptorpositioninrequired_contactsin,auto(vs)")
    
    
    
    auto_generated_contact_pairs = None
    receptor_msa_file_for_fake = None  
    binder_msa_file_for_fake = None  
    binder_template_modified = None  
    if False and args.required_contacts:  
        
        receptor_positions_str = args.required_contacts.split(',')
        receptor_indices_raw = [int(pos.strip()) - 1 for pos in receptor_positions_str if pos.strip()]  
        
        
        if fixed_receptor_positions:
            receptor_indices = [idx for idx in receptor_indices_raw if idx not in fixed_receptor_positions]
            excluded_indices = [idx for idx in receptor_indices_raw if idx in fixed_receptor_positions]
            if excluded_indices:
                excluded_indices_1based = [idx + 1 for idx in excluded_indices]
                print(f"\n{'='*60}")
                print(f" autovsRequired Contacts")
                print(f"{'='*60}")
                print(f"Receptorresidueposition(1-based,input): {args.required_contacts}")
                print(f" toandfixedresiduevsreceptorposition: {excluded_indices_1based} (1-based)")
                print(f" thispositionalreadyinfixed_residue_contactsin,fromrequired_contactsin")
                print(f" receptorposition: {[idx + 1 for idx in receptor_indices]} (1-based)")
            else:
                receptor_indices = receptor_indices_raw
                print(f"\n{'='*60}")
                print(f" autovsRequired Contacts")
                print(f"{'='*60}")
                print(f"Receptorresidueposition(1-based,input): {args.required_contacts}")
                print(f"convertas0-basedindex(use): {receptor_indices}")
        else:
            receptor_indices = receptor_indices_raw
            print(f"\n{'='*60}")
            print(f" autovsRequired Contacts")
            print(f"{'='*60}")
            print(f"Receptorresidueposition(1-based,input): {args.required_contacts}")
            print(f"convertas0-basedindex(use): {receptor_indices}")
        
        
        if not receptor_indices:
            print(f"\n warning:allreceptorpositionalreadyinfixed_residue_contactsin,skipautovs")
            auto_generated_contact_pairs = None
        else:
        
            
            if not cdr_regions:
                print(f" warning:notCDRregion(need--nanobodyor--fabmode,or--cdr_regions)")
                print(f" noautovs,skiprequired_contacts")
                auto_generated_contact_pairs = None
            else:
                
                if nanobody_mode:
                    binder_seq_template = args.binder_template  
                elif fab_mode:
                    binder_seq_template, _, _ = get_fab_template(args.fab_template)
                else:
                    
                    binder_seq_template = "X" * binder_length
                
                
                auto_generated_contact_pairs, binder_aa_map = auto_pair_receptor_to_cdr(
                    receptor_seq=receptor_seq,
                    receptor_indices=receptor_indices,
                    cdr_regions=cdr_regions,
                    binder_length=len(binder_seq_template),
                    exclude_binder_positions=fixed_binder_positions,  
                    verbose=True
                )
                
                if auto_generated_contact_pairs:
                    print(f"\n autovsok: {len(auto_generated_contact_pairs)} contactvs")
                    print(f" contactvs (receptor_idx, binder_idx,1-based):")
                    for r_idx, b_idx in auto_generated_contact_pairs:
                        print(f" Receptorresidue {r_idx+1} (1-based) <-> Binderresidue {b_idx+1} (1-based) (binder: {binder_aa_map.get(b_idx, '?')})")
                    
                    
                    
                    
                    print(f"\n bindersequence...")
                    
                    
                    binder_seq_for_yaml = list(binder_seq_template)
                    
                    if fixed_binder_aa_map:
                        for binder_idx, aa_type in fixed_binder_aa_map.items():
                            if 0 <= binder_idx < len(binder_seq_for_yaml):
                                binder_seq_for_yaml[binder_idx] = aa_type
                                print(f" Binderposition {binder_idx+1} (1-based): {binder_seq_template[binder_idx]} -> {aa_type} (fixedresidue,forYAML)")
                    
                    for binder_idx, aa_type in binder_aa_map.items():
                        if 0 <= binder_idx < len(binder_seq_for_yaml):
                            binder_seq_for_yaml[binder_idx] = aa_type
                            print(f" Binderposition {binder_idx+1} (1-based): {binder_seq_template[binder_idx]} -> {aa_type} (contactvs,forYAML)")
                    
                    binder_seq_for_yaml_str = ''.join(binder_seq_for_yaml)
                    
                    
                    binder_seq_for_msa = list(binder_seq_for_yaml_str)  
                    x_count = 0
                    for i in range(len(binder_seq_for_msa)):
                        if binder_seq_for_msa[i] == 'X':
                            binder_seq_for_msa[i] = 'A'
                            x_count += 1
                    
                    binder_seq_for_msa_str = ''.join(binder_seq_for_msa)
                    
                    if x_count > 0:
                        print(f" {x_count} XpositionalreadyasA(forfake MSA,fake_msa_generatornotsupportX)")
                    
                    print(f"\n YAMLsequence(for):")
                    print(f" length: {len(binder_seq_for_yaml_str)}")
                    print(f" contactvspositionresidue: {[binder_seq_for_yaml_str[b_idx] for _, b_idx in auto_generated_contact_pairs]}")
                    print(f" contactvspositionkeepX: {binder_seq_for_yaml_str.count('X')} ")
                    print(f"\n MSAsequence(forgeneratefake MSA):")
                    print(f" length: {len(binder_seq_for_msa_str)}")
                    print(f" allXalreadyasA")
                    
                    
                    
                    binder_template_modified = binder_seq_for_yaml_str
                    
                    
                    print(f"\ngeneratefake MSA...")
                    try:
                        from dream_boltz2.msa.fake_msa import generate_fake_msa_mode2
                        
                        
                        temp_msa_dir = get_fake_msa_dir_from_args(args)
                        if "fake_msa" in str(temp_msa_dir):
                            print(f" Fake MSA saveto: {temp_msa_dir.absolute()}")
                        
                        
                        all_contact_pairs = fixed_contact_pairs.copy() if fixed_residue_contacts else []
                        if auto_generated_contact_pairs:
                            all_contact_pairs.extend(auto_generated_contact_pairs)
                        
                        print(f"\n contactvslist(forfake MSA):")
                        print(f" fixedcontactvs: {len(fixed_contact_pairs) if fixed_residue_contacts else 0} ")
                        print(f" autovs: {len(auto_generated_contact_pairs) if auto_generated_contact_pairs else 0} ")
                        print(f": {len(all_contact_pairs)} ")
                        
                        
                        receptor_df, binder_df = generate_fake_msa_mode2(
                            receptor_seq=receptor_seq,
                            binder_seq=binder_seq_for_msa_str,  
                            contact_pairs=all_contact_pairs,  
                            mapping_mode="complementary",
                            num_seqs=args.max_msa_seqs,  
                            correlation_strength=0.9,  
                            output_dir=temp_msa_dir,
                            verbose=True
                        )
                        
                        
                        receptor_msa_path = temp_msa_dir / "receptor_msa.csv"
                        binder_msa_path = temp_msa_dir / "binder_msa.csv"
                        receptor_df.to_csv(receptor_msa_path, index=False)
                        binder_df.to_csv(binder_msa_path, index=False)
                        
                        
                        receptor_msa_file_for_fake = str(receptor_msa_path)
                        binder_msa_file_for_fake = str(binder_msa_path)
                        print(f" Fake MSAalreadygenerate(vsMSA):")
                        print(f"   Receptor MSA: {receptor_msa_file_for_fake}")
                        print(f"   Binder MSA: {binder_msa_file_for_fake}")
                        print(f" requiredusevsreceptorandbinder MSAkeepvs！")
                        
                    except Exception as e:
                        print(f" generatefake MSAfailed: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f" useemptyMSA")
                else:
                    print(f" warning:autovsfailed,notusefake MSA")
    
    
    if not args.required_contacts and fixed_residue_contacts:
        print(f"\n{'='*60}")
        print(f" useFixed Residue ContactsgenerateFake MSA")
        print(f"{'='*60}")
        
        
        if nanobody_mode:
            binder_seq_template = args.binder_template  
        elif fab_mode:
            binder_seq_template, _, _ = get_fab_template(args.fab_template)
        else:
            binder_seq_template = "X" * binder_length
        
        
        binder_seq_for_yaml = list(binder_seq_template)
        for binder_idx, aa_type in fixed_binder_aa_map.items():
            if 0 <= binder_idx < len(binder_seq_for_yaml):
                binder_seq_for_yaml[binder_idx] = aa_type
        
        binder_seq_for_yaml_str = ''.join(binder_seq_for_yaml)
        
        
        binder_seq_for_msa = list(binder_seq_for_yaml_str)
        for i in range(len(binder_seq_for_msa)):
            if binder_seq_for_msa[i] == 'X':
                binder_seq_for_msa[i] = 'A'
        binder_seq_for_msa_str = ''.join(binder_seq_for_msa)
        
        binder_template_modified = binder_seq_for_yaml_str
        
        
        print(f"\ngeneratefake MSA(fixedvs)...")
        try:
            from dream_boltz2.msa.fake_msa import generate_fake_msa_mode2
            
            
            temp_msa_dir = get_fake_msa_dir_from_args(args)
            if "fake_msa" in str(temp_msa_dir):
                print(f" Fake MSA saveto: {temp_msa_dir.absolute()}")
            
            receptor_df, binder_df = generate_fake_msa_mode2(
                receptor_seq=receptor_seq,
                binder_seq=binder_seq_for_msa_str,
                contact_pairs=fixed_contact_pairs,  
                mapping_mode="complementary",
                num_seqs=args.max_msa_seqs,
                correlation_strength=0.9,
                output_dir=temp_msa_dir,
                verbose=True
            )
            
            receptor_msa_path = temp_msa_dir / "receptor_msa.csv"
            binder_msa_path = temp_msa_dir / "binder_msa.csv"
            receptor_df.to_csv(receptor_msa_path, index=False)
            binder_df.to_csv(binder_msa_path, index=False)
            
            
            receptor_msa_file_for_fake = str(receptor_msa_path)
            binder_msa_file_for_fake = str(binder_msa_path)
            print(f" Fake MSAalreadygenerate(vsMSA):")
            print(f"   Receptor MSA: {receptor_msa_file_for_fake}")
            print(f"   Binder MSA: {binder_msa_file_for_fake}")
            print(f" requiredusevsreceptorandbinder MSAkeepvs！")
            
        except Exception as e:
            print(f" generatefake MSAfailed: {e}")
            import traceback
            traceback.print_exc()
    
    
    contact_constraints = None
    if args.contact_constraints:
        
        
        contact_constraints = []
        for constraint_str in args.contact_constraints.split(','):
            parts = constraint_str.strip().split(':')
            if len(parts) == 3:
                
                binder_idx_1based = int(parts[0])
                receptor_idx_1based = int(parts[1])
                binder_idx = binder_idx_1based - 1  
                receptor_idx = receptor_idx_1based - 1  
                distance = float(parts[2])
                contact_constraints.append((binder_idx, receptor_idx, distance))
            else:
                print(f" warning:nocontact constraint: {constraint_str},as 'binder_idx:receptor_idx:distance'")
        print(f"\n Contactconstraint(Boltz2): {len(contact_constraints)} constraint")
        print(f" note:constraintinindexinBoltz2convertas0-based")
        for binder_idx, receptor_idx, distance in contact_constraints:
            print(f" Binderresidue {binder_idx+1} (1-based) <-> receptorresidue {receptor_idx+1} (1-based) (distance ≤ {distance} Å)")
        print(f" thisconstraintto(s,z)optimizationin(Boltz2constraint features)")
    
    
    ligands = None
    if args.ligand:
        ligands = []
        
        ligand_specs = args.ligand.split(',')
        for spec in ligand_specs:
            spec = spec.strip()
            
            
            parts = spec.split(':')
            if len(parts) == 2:
                
                ligand_id = parts[0].strip()
                ccd_code = parts[1].strip()
                ligands.append({'id': ligand_id, 'ccd': ccd_code})
                print(f" addligand: chainID={ligand_id}, CCD={ccd_code}")
            elif len(parts) >= 3 and parts[1].lower() == 'smiles':
                
                ligand_id = parts[0].strip()
                smiles_str = ':'.join(parts[2:])  
                ligands.append({'id': ligand_id, 'smiles': smiles_str})
                print(f" addligand: chainID={ligand_id}, SMILES={smiles_str[:50]}...")
            else:
                print(f" warning:noligand: {spec},as 'id:ccd' or 'id:smiles:SMILES_STRING'")
        if len(ligands) > 0:
            print(f"\n ligandconfig: {len(ligands)} ligand")
    
    
    # =========================================================================
    
    # =========================================================================
    yaml_receptor_seq = None
    yaml_binder_seq = None
    yaml_receptor_msa = None
    yaml_binder_modifications = []
    
    if hasattr(args, '_dream_config') and args._dream_config is not None:
        config = args._dream_config
        
        
        for seq_item in config.sequences:
            if seq_item.chain_id in config.receptor_chains:
                yaml_receptor_seq = seq_item.sequence
                yaml_receptor_msa = seq_item.msa
                print(f" [YAML] Receptorsequence (chain {seq_item.chain_id}): {len(yaml_receptor_seq)} residue")
                if yaml_receptor_msa:
                    print(f" [YAML] Receptor MSA: {yaml_receptor_msa}")
            elif seq_item.chain_id in config.binder_chains:
                yaml_binder_seq = seq_item.sequence
                yaml_binder_modifications = seq_item.modifications
                print(f" [YAML] Bindersequence (chain {seq_item.chain_id}): {len(yaml_binder_seq)} residue")
                if yaml_binder_modifications:
                    print(f" [YAML] Binder modifications: {yaml_binder_modifications}")
        
        
        if yaml_receptor_msa and not args.receptor_msa:
            args.receptor_msa = yaml_receptor_msa
            print(f" [YAML] use YAML in Receptor MSA: {args.receptor_msa}")
        
        
        if yaml_binder_seq and not args.binder_template:
            args.binder_template = yaml_binder_seq
            print(f" [YAML] use YAML in Binder sequenceastemplate")
        
        
        yaml_template = config.get_first_template()
        if yaml_template and not args.template_file:
            args.template_file = yaml_template.path
            if yaml_template.chain_ids:
                args.template_chain_id = yaml_template.chain_ids[0]
            
            args.template_force = yaml_template.force
            args.template_threshold = yaml_template.threshold
            print(f" [YAML] Templateconfig:")
            print(f" file: {args.template_file} ({yaml_template.file_type})")
            print(f"   Chain ID: {args.template_chain_id}")
            if yaml_template.template_ids:
                print(f"   Template ID: {yaml_template.template_ids}")
            print(f"   Force: {args.template_force}, Threshold: {args.template_threshold} A")
        
        
        if config.msa.fake_msa_enabled:
            if config.msa.fake_msa_has_contacts:
                print(f" [YAML] Fake MSA: contactvsmode ({len(config.msa.fake_msa)} vscontactresidue)")
                for item in config.msa.fake_msa:
                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                        print(f"   {item[0]}:{item[1]} <-> {item[2]}:{item[3]}")
            else:
                print(f" [YAML] Fake MSA: automode ( hotspot_indices generate)")
    
    if yaml_receptor_seq:
        
        print(f"\n use YAML configin Receptor sequence")
        receptor_seq_original = yaml_receptor_seq
    elif args.pdb:
        
        complex_info = parse_pdb_complex(args.pdb, args.receptor_chain, 'A', verbose=True)
        receptor_seq_original = complex_info['receptor_sequence']  
    else:
        receptor_seq_original = None
    
    if receptor_seq_original is None:
        print(' error:required receptor sequence( YAML configor --pdb argument)')
        return
    
    
    
    receptor_seq = receptor_seq_original  
    original_receptor_res_names = None  
    
    if args.anti_contact_residues:
        print(f"\n Anti-Contactmutationalready:")
        print(f" region: {args.anti_contact_residues}")
        print(f" mutation: {args.anti_contact_mutation_type}")
        
        
        mutation_indices = parse_residue_ranges(args.anti_contact_residues)
        
        if mutation_indices:
            
            valid_indices = [idx for idx in mutation_indices if 0 <= idx < len(receptor_seq_original)]
            if len(valid_indices) < len(mutation_indices):
                invalid_indices = [idx for idx in mutation_indices if idx not in valid_indices]
                invalid_indices_1based = [idx + 1 for idx in invalid_indices]  
                print(f" warning:residueindex,alreadyskip: {invalid_indices_1based} (1-based)")
            
            if valid_indices:
                
                from boltz.data.const import prot_letter_to_token
                original_receptor_res_names = []
                for idx, aa in enumerate(receptor_seq_original):
                    if idx in valid_indices:
                        
                        original_receptor_res_names.append(prot_letter_to_token.get(aa, 'UNK'))
                    else:
                        
                        original_receptor_res_names.append(prot_letter_to_token.get(aa, 'UNK'))
                
                
                receptor_seq = apply_anti_contact_mutation(
                    receptor_seq_original, 
                    valid_indices, 
                    args.anti_contact_mutation_type
                )
                
                
                mutated_positions = sorted(valid_indices)
                mutated_positions_1based = [idx + 1 for idx in mutated_positions]  
                print(f" okmutation {len(valid_indices)} residue:")
                print(f" mutationposition(1-based): {mutated_positions_1based[:10]}{'...' if len(mutated_positions) > 10 else ''}")
                print(f" sequence: {receptor_seq_original[mutated_positions[0]:mutated_positions[0]+5]}...")
                print(f" mutation: {receptor_seq[mutated_positions[0]:mutated_positions[0]+5]}...")
                print(f" usemutationsequence(N),PDBoutputassequence")
        else:
            print(f" warning:nottoresidueindex,skipmutation")
    
    # =========================================================================
    
    # =========================================================================
    
    
    if args.fixed_residue_contacts and not fixed_residue_contacts_parsed:
        print(f"\n{'='*60}")
        print(f" parseFixed Residue Contacts(fixedresiduevs)")
        print(f"{'='*60}")
        
        fixed_residue_contacts, fixed_binder_positions, fixed_receptor_positions, fixed_binder_aa_map, fixed_contact_pairs = parse_fixed_residue_contacts(
            args.fixed_residue_contacts,
            receptor_seq,
            cdr_regions if cdr_regions else [],
            binder_length,
            verbose=True
        )
        
        print(f"\n fixedresidue:")
        print(f" fixedvs: {len(fixed_residue_contacts)}")
        print(f" fixedbinderposition: {sorted([idx + 1 for idx in fixed_binder_positions])} (1-based)")
        print(f" fixedreceptorposition: {sorted([idx + 1 for idx in fixed_receptor_positions])} (1-based)")
        print(f" thispositioninsequenceoptimizationnotMPNN")
        print(f" thisvsinfake MSAin")
        print(f" ifthisreceptorpositioninrequired_contactsin,auto(vs)")
        fixed_residue_contacts_parsed = True
    
    
    if args.required_contacts and auto_generated_contact_pairs is None:
        
        receptor_positions_str = args.required_contacts.split(',')
        receptor_indices_raw = [int(pos.strip()) - 1 for pos in receptor_positions_str if pos.strip()]  
        
        
        if fixed_receptor_positions:
            receptor_indices = [idx for idx in receptor_indices_raw if idx not in fixed_receptor_positions]
            excluded_indices = [idx for idx in receptor_indices_raw if idx in fixed_receptor_positions]
            if excluded_indices:
                excluded_indices_1based = [idx + 1 for idx in excluded_indices]
                print(f"\n{'='*60}")
                print(f" autovsRequired Contacts")
                print(f"{'='*60}")
                print(f"Receptorresidueposition(1-based,input): {args.required_contacts}")
                print(f" toandfixedresiduevsreceptorposition: {excluded_indices_1based} (1-based)")
                print(f" thispositionalreadyinfixed_residue_contactsin,fromrequired_contactsin")
                print(f" receptorposition: {[idx + 1 for idx in receptor_indices]} (1-based)")
            else:
                receptor_indices = receptor_indices_raw
                print(f"\n{'='*60}")
                print(f" autovsRequired Contacts")
                print(f"{'='*60}")
                print(f"Receptorresidueposition(1-based,input): {args.required_contacts}")
                print(f"convertas0-basedindex(use): {receptor_indices}")
        else:
            receptor_indices = receptor_indices_raw
            print(f"\n{'='*60}")
            print(f" autovsRequired Contacts")
            print(f"{'='*60}")
            print(f"Receptorresidueposition(1-based,input): {args.required_contacts}")
            print(f"convertas0-basedindex(use): {receptor_indices}")
        
        
        if not receptor_indices:
            print(f"\n warning:allreceptorpositionalreadyinfixed_residue_contactsin,skipautovs")
            auto_generated_contact_pairs = None
        else:
            
            if not cdr_regions:
                print(f" warning:notCDRregion(need--nanobodyor--fabmode,or--cdr_regions)")
                print(f" noautovs,skiprequired_contacts")
                auto_generated_contact_pairs = None
            else:
                
                if nanobody_mode:
                    binder_seq_template = args.binder_template  
                elif fab_mode:
                    binder_seq_template, _, _ = get_fab_template(args.fab_template)
                else:
                    
                    binder_seq_template = "X" * binder_length
                
                
                auto_generated_contact_pairs, binder_aa_map = auto_pair_receptor_to_cdr(
                    receptor_seq=receptor_seq,
                    receptor_indices=receptor_indices,
                    cdr_regions=cdr_regions,
                    binder_length=len(binder_seq_template),
                    exclude_binder_positions=fixed_binder_positions,  
                    verbose=True
                )
                
                if auto_generated_contact_pairs:
                    print(f"\n autovsok: {len(auto_generated_contact_pairs)} contactvs")
                    print(f" contactvs (receptor_idx, binder_idx,1-based):")
                    for r_idx, b_idx in auto_generated_contact_pairs:
                        print(f" Receptorresidue {r_idx+1} (1-based) <-> Binderresidue {b_idx+1} (1-based) (binder: {binder_aa_map.get(b_idx, '?')})")
                    
                    
                    
                    
                    print(f"\n bindersequence...")
                    
                    
                    binder_seq_for_yaml = list(binder_seq_template)
                    
                    if fixed_binder_aa_map:
                        for binder_idx, aa_type in fixed_binder_aa_map.items():
                            if 0 <= binder_idx < len(binder_seq_for_yaml):
                                binder_seq_for_yaml[binder_idx] = aa_type
                                print(f" Binderposition {binder_idx+1} (1-based): {binder_seq_template[binder_idx]} -> {aa_type} (fixedresidue,forYAML)")
                    
                    for binder_idx, aa_type in binder_aa_map.items():
                        if 0 <= binder_idx < len(binder_seq_for_yaml):
                            binder_seq_for_yaml[binder_idx] = aa_type
                            print(f" Binderposition {binder_idx+1} (1-based): {binder_seq_template[binder_idx]} -> {aa_type} (contactvs,forYAML)")
                    
                    binder_seq_for_yaml_str = ''.join(binder_seq_for_yaml)
                    
                    
                    binder_seq_for_msa = list(binder_seq_for_yaml_str)  
                    x_count = 0
                    for i in range(len(binder_seq_for_msa)):
                        if binder_seq_for_msa[i] == 'X':
                            binder_seq_for_msa[i] = 'A'
                            x_count += 1
                    
                    binder_seq_for_msa_str = ''.join(binder_seq_for_msa)
                    
                    if x_count > 0:
                        print(f" {x_count} XpositionalreadyasA(forfake MSA,fake_msa_generatornotsupportX)")
                    
                    print(f"\n YAMLsequence(for):")
                    print(f" length: {len(binder_seq_for_yaml_str)}")
                    print(f" contactvspositionresidue: {[binder_seq_for_yaml_str[b_idx] for _, b_idx in auto_generated_contact_pairs]}")
                    print(f" contactvspositionkeepX: {binder_seq_for_yaml_str.count('X')} ")
                    print(f"\n MSAsequence(forgeneratefake MSA):")
                    print(f" length: {len(binder_seq_for_msa_str)}")
                    print(f" allXalreadyasA")
                    
                    
                    
                    binder_template_modified = binder_seq_for_yaml_str
                    
                    
                    print(f"\ngeneratefake MSA...")
                    try:
                        from dream_boltz2.msa.fake_msa import generate_fake_msa_mode2
                        
                        
                        temp_msa_dir = get_fake_msa_dir_from_args(args)
                        if "fake_msa" in str(temp_msa_dir):
                            print(f" Fake MSA saveto: {temp_msa_dir.absolute()}")
                        
                        
                        all_contact_pairs = fixed_contact_pairs.copy() if fixed_residue_contacts else []
                        if auto_generated_contact_pairs:
                            all_contact_pairs.extend(auto_generated_contact_pairs)
                        
                        print(f"\n contactvslist(forfake MSA):")
                        print(f" fixedcontactvs: {len(fixed_contact_pairs) if fixed_residue_contacts else 0} ")
                        print(f" autovs: {len(auto_generated_contact_pairs) if auto_generated_contact_pairs else 0} ")
                        print(f": {len(all_contact_pairs)} ")
                        
                        
                        receptor_df, binder_df = generate_fake_msa_mode2(
                            receptor_seq=receptor_seq,
                            binder_seq=binder_seq_for_msa_str,  
                            contact_pairs=all_contact_pairs,  
                            mapping_mode="complementary",
                            num_seqs=args.max_msa_seqs,  
                            correlation_strength=0.9,  
                            output_dir=temp_msa_dir,
                            verbose=True
                        )
                        
                        
                        receptor_msa_path = temp_msa_dir / "receptor_msa.csv"
                        binder_msa_path = temp_msa_dir / "binder_msa.csv"
                        receptor_df.to_csv(receptor_msa_path, index=False)
                        binder_df.to_csv(binder_msa_path, index=False)
                        
                        
                        receptor_msa_file_for_fake = str(receptor_msa_path)
                        binder_msa_file_for_fake = str(binder_msa_path)
                        print(f" Fake MSAalreadygenerate(vsMSA):")
                        print(f"   Receptor MSA: {receptor_msa_file_for_fake}")
                        print(f"   Binder MSA: {binder_msa_file_for_fake}")
                        print(f" requiredusevsreceptorandbinder MSAkeepvs！")
                        
                    except Exception as e:
                        print(f" generatefake MSAfailed: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f" useemptyMSA")
                else:
                    print(f" warning:autovsfailed,notusefake MSA")
    
    
    if not args.required_contacts and fixed_residue_contacts and not binder_msa_file_for_fake:
        print(f"\n{'='*60}")
        print(f" useFixed Residue ContactsgenerateFake MSA")
        print(f"{'='*60}")
        try:
            from dream_boltz2.msa.fake_msa import generate_fake_msa_mode2
            
            
            temp_msa_dir = get_fake_msa_dir_from_args(args)
            if "fake_msa" in str(temp_msa_dir):
                print(f" Fake MSA saveto: {temp_msa_dir.absolute()}")
            
            
            if nanobody_mode:
                binder_seq_template = args.binder_template  
            elif fab_mode:
                binder_seq_template, _, _ = get_fab_template(args.fab_template)
            else:
                binder_seq_template = "X" * binder_length
            
            
            binder_seq_for_msa = list(binder_seq_template)
            x_count = 0
            for i in range(len(binder_seq_for_msa)):
                if binder_seq_for_msa[i] == 'X':
                    binder_seq_for_msa[i] = 'A'
                    x_count += 1
            binder_seq_for_msa_str = ''.join(binder_seq_for_msa)
            
            
            temp_msa_dir = get_fake_msa_dir_from_args(args)
            if "fake_msa" in str(temp_msa_dir):
                print(f" Fake MSA saveto: {temp_msa_dir.absolute()}")
            
            
            receptor_df, binder_df = generate_fake_msa_mode2(
                receptor_seq=receptor_seq,
                binder_seq=binder_seq_for_msa_str,
                contact_pairs=fixed_contact_pairs,
                mapping_mode="complementary",
                num_seqs=args.max_msa_seqs,
                correlation_strength=0.9,
                output_dir=temp_msa_dir,
                verbose=True
            )
            
            
            receptor_msa_path = temp_msa_dir / "receptor_msa.csv"
            binder_msa_path = temp_msa_dir / "binder_msa.csv"
            receptor_df.to_csv(receptor_msa_path, index=False)
            binder_df.to_csv(binder_msa_path, index=False)
            
            receptor_msa_file_for_fake = str(receptor_msa_path)
            binder_msa_file_for_fake = str(binder_msa_path)
            print(f" Fake MSAalreadygenerate(fixedcontactvs):")
            print(f"   Receptor MSA: {receptor_msa_file_for_fake}")
            print(f"   Binder MSA: {binder_msa_file_for_fake}")
        except Exception as e:
            print(f" generatefake MSAfailed: {e}")
            import traceback
            traceback.print_exc()
    
    
    
    
    if (hasattr(args, '_dream_config') and args._dream_config is not None
            and args._dream_config.msa.fake_msa_enabled
            and not binder_msa_file_for_fake):
        _config = args._dream_config
        
        
        _contact_pairs_for_msa = []
        
        
        if _config.msa.fake_msa_has_contacts:
            _contact_pairs_for_msa = _config.msa.get_contact_pairs(
                receptor_chains=_config.receptor_chains,
                binder_chains=_config.binder_chains
            )
            if _contact_pairs_for_msa:
                print(f"\n{'='*60}")
                print(f" [YAML] fake_msa: contactvsmode -- {len(_contact_pairs_for_msa)} vscontactresidue")
                print(f"{'='*60}")
                for item in _config.msa.fake_msa:
                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                        print(f"   {item[0]}:{item[1]} <-> {item[2]}:{item[3]}")
                print(f" parse (receptor_0based, binder_0based): {_contact_pairs_for_msa}")
            else:
                print(f" [YAML] fake_msa contactvsparseasempty,checkchain ID whether")
        
        
        else:
            _hotspot_list = _config.hotspot_indices if _config.hotspot_indices else []
            if _hotspot_list:
                _binder_seq_raw = args.binder_template if args.binder_template else "X" * binder_length
                _binder_mid = len(_binder_seq_raw) // 2
                for i, hs in enumerate(_hotspot_list):
                    receptor_idx = hs - 1  # 1-based -> 0-based
                    binder_idx = min(_binder_mid + i, len(_binder_seq_raw) - 1)
                    _contact_pairs_for_msa.append((receptor_idx, binder_idx))
                print(f"\n{'='*60}")
                print(f" [YAML] fake_msa: true -- hotspot_indices autogenerate")
                print(f" Hotspotresidue (1-based): {_hotspot_list}")
                print(f" autocontactvs (receptor_0based, binder_0based): {_contact_pairs_for_msa}")
                print(f"{'='*60}")
            else:
                print(f" [YAML] fake_msa: true,no hotspot_indices orcontactvs,nogenerate")
        
        
        if _contact_pairs_for_msa:
            try:
                from dream_boltz2.msa.fake_msa import generate_fake_msa_mode2
                
                
                _binder_seq_raw = args.binder_template if args.binder_template else "X" * binder_length
                _binder_seq_msa = ''.join('A' if c == 'X' else c for c in _binder_seq_raw)
                
                
                _msa_dir = get_fake_msa_dir_from_args(args)
                if "fake_msa" in str(_msa_dir):
                    print(f" Fake MSA saveto: {_msa_dir.absolute()}")
                else:
                    print(f" notoutput directory,usedirectory: {_msa_dir}")
                
                _receptor_df, _binder_df = generate_fake_msa_mode2(
                    receptor_seq=receptor_seq,
                    binder_seq=_binder_seq_msa,
                    contact_pairs=_contact_pairs_for_msa,
                    mapping_mode="complementary",
                    num_seqs=args.max_msa_seqs,
                    correlation_strength=0.9,
                    output_dir=_msa_dir,
                    verbose=True
                )
                
                _receptor_msa_path = _msa_dir / "receptor_msa.csv"
                _binder_msa_path = _msa_dir / "binder_msa.csv"
                _receptor_df.to_csv(_receptor_msa_path, index=False)
                _binder_df.to_csv(_binder_msa_path, index=False)
                
                receptor_msa_file_for_fake = str(_receptor_msa_path)
                binder_msa_file_for_fake = str(_binder_msa_path)
                print(f" [YAML] Fake MSA alreadygenerate:")
                print(f"   Receptor MSA: {receptor_msa_file_for_fake}")
                print(f"   Binder MSA: {binder_msa_file_for_fake}")
            except Exception as e:
                print(f" [YAML] generate fake MSA failed: {e}")
                import traceback
                traceback.print_exc()
        
    
    
    import inspect
    sig = inspect.signature(feature_prep.prepare_complex_features)
    has_binder_chains = 'binder_chains' in sig.parameters
    has_binder_template = 'binder_template' in sig.parameters
    has_binder_msa_file = 'binder_msa_file' in sig.parameters
    has_binder_modifications = 'binder_modifications' in sig.parameters
    has_ligands = 'ligands' in sig.parameters
    has_template_file = 'template_file' in sig.parameters
    
    
    binder_chains = None
    if yaml_binder_seq:
        if not args.binder_template:
            args.binder_template = yaml_binder_seq
            print(f" [YAML] use YAML in Binder sequenceastemplate")
        binder_length = len(yaml_binder_seq)
        print(f" [YAML] Binder length: {binder_length}")
    
        
        if fab_mode and fab_chain_info:
            
            
            if binder_template_modified is not None:
                
                full_seq = binder_template_modified
                h_chain_seq = full_seq[:fab_chain_info['H_chain_length']]
                l_chain_seq = full_seq[fab_chain_info['H_chain_length']:]
                print(f"[INFO] Fabmode:usesequence(required_contacts)")
            else:
                
                h_chain_seq = fab_sequence[:fab_chain_info['H_chain_length']]
                l_chain_seq = fab_sequence[fab_chain_info['H_chain_length']:]
                print(f"[INFO] Fabmode:usesequence")
            
            
            binder_chains = [('C', h_chain_seq), ('D', l_chain_seq)]  
            binder_length = fab_chain_info['H_chain_length'] + fab_chain_info['L_chain_length']
            print(f" Hchain (C): {len(h_chain_seq)} residue")
            print(f" Lchain (D): {len(l_chain_seq)} residue")
            print(f" length: {binder_length}")
        elif args.binder_template:
            
            binder_length = len(args.binder_template)
            print(f"[INFO] usebinder_templatelengthasbinder_length: {binder_length}")
    elif args.pdb and 'complex_info' in locals():
        
        binder_length = len(complex_info['binder_sequence']) if complex_info.get('binder_sequence') else 0
        if binder_length == 0:
            print(f" warning:PDBinnottobinderchain,notbinder_template,usedefaultlength60")
            binder_length = 60
    else:
        if not yaml_binder_seq:
            print(f" warning:notbindersequence,usedefaultlength60")
            binder_length = 60
    
    
    prepare_kwargs = {
        'receptor_sequence': receptor_seq,
        'binder_length': binder_length,
        'binder_cyclic': args.cyclic,
        'receptor_msa_file': args.receptor_msa,
        'max_msa_seqs': args.max_msa_seqs,
    }
        
    
    if has_binder_modifications and yaml_binder_modifications:
        prepare_kwargs['binder_modifications'] = yaml_binder_modifications
        print(f" [YAML] Binder modifications: {len(yaml_binder_modifications)} ")
        for mod in yaml_binder_modifications:
            print(f" - position {mod['position']}: {mod['ccd']}")
    
    
    if binder_msa_file_for_fake and has_binder_msa_file:
        prepare_kwargs['binder_msa_file'] = binder_msa_file_for_fake
        print(f"\n useautogeneratefake Binder MSA: {binder_msa_file_for_fake}")
    if receptor_msa_file_for_fake:
        
        if args.receptor_msa and args.receptor_msa != receptor_msa_file_for_fake:
            print(f" warning:fake MSA already, receptor_msa keepvs")
        prepare_kwargs['receptor_msa_file'] = receptor_msa_file_for_fake
        print(f" useautogeneratefake Receptor MSA: {receptor_msa_file_for_fake}")
    
    if has_binder_chains and binder_chains is not None:
        
        prepare_kwargs['binder_chains'] = binder_chains
    elif has_binder_template:
        
        
        if binder_template_modified is not None:
            prepare_kwargs['binder_template'] = binder_template_modified
            print(f"\n usebindersequence(contactvspositionalready,positionkeepX)")
        elif args.binder_template:
            prepare_kwargs['binder_template'] = args.binder_template
    
    if has_ligands and ligands:
        prepare_kwargs['ligands'] = ligands
        print(f" ligandstoprepare_complex_features: {len(ligands)} ligand")
        for lig in ligands:
            print(f"   - {lig}")
    elif has_ligands:
        print(f" has_ligands=Trueligands=Noneorempty,notligandsargument")
    
    
    
    if has_template_file and args.template_file and args.template_chain_id:
        prepare_kwargs['template_file'] = args.template_file
        prepare_kwargs['template_chain_id'] = args.template_chain_id
        prepare_kwargs['template_force'] = args.template_force
        prepare_kwargs['template_threshold'] = args.template_threshold
        
        if (hasattr(args, '_dream_config') and args._dream_config is not None
                and args._dream_config.templates):
            _yaml_tmpl = args._dream_config.get_first_template()
            if _yaml_tmpl and _yaml_tmpl.template_ids:
                has_template_id = 'template_id' in sig.parameters
                if has_template_id:
                    prepare_kwargs['template_id'] = _yaml_tmpl.template_ids[0]
        print(f"\n Templateconstraintalready:")
        print(f" Templatefile: {args.template_file}")
        print(f" TemplatechainID: {args.template_chain_id}")
        print(f" Forceconstraint: {args.template_force}")
        print(f"  Threshold: {args.template_threshold} Å")
    
    
    if 'yaml_constraints' in locals() and yaml_constraints:
        
        import inspect
        sig = inspect.signature(feature_prep.prepare_complex_features)
        has_yaml_constraints = 'yaml_constraints' in sig.parameters
        if has_yaml_constraints:
            prepare_kwargs['yaml_constraints'] = yaml_constraints
            print(f" YAMLconstrainttoprepare_complex_features: {len(yaml_constraints)} constraint")
        else:
            print(f" prepare_complex_featuresnotsupportyaml_constraintsargument,constraintnot")
    
    
    
    sig = inspect.signature(feature_prep.prepare_complex_features)
    has_receptor_chain_id = 'receptor_chain_id' in sig.parameters
    has_binder_chain_id = 'binder_chain_id' in sig.parameters
    if has_receptor_chain_id and config.receptor_chains and len(config.receptor_chains) > 0:
        
        prepare_kwargs['receptor_chain_id'] = config.receptor_chains[0]
        print(f" useYAMLconfiginreceptorchainID: {config.receptor_chains[0]}")
    if has_binder_chain_id and config.binder_chains and len(config.binder_chains) > 0:
        
        
        if binder_chains is None or len(binder_chains) == 0:
            prepare_kwargs['binder_chain_id'] = config.binder_chains[0]
            print(f" useYAMLconfiginbinderchainID: {config.binder_chains[0]}")
        else:
            print(f" chainmode:binderchainIDalreadyinbinder_chainsin")
    
    if has_binder_chains or has_binder_template or has_binder_modifications or has_ligands or (has_template_file and args.template_file):
        base_feats, metadata = feature_prep.prepare_complex_features(**prepare_kwargs)
    else:
        
        print(' warning:currentprepare_complex_featuresnotsupportargument,usedefault')
        base_feats, metadata = feature_prep.prepare_complex_features(
            receptor_sequence=receptor_seq,
            binder_length=binder_length,
            binder_cyclic=args.cyclic,
            receptor_msa_file=args.receptor_msa,
            max_msa_seqs=args.max_msa_seqs
        )
    
    
    if 'original_receptor_res_names' in locals() and original_receptor_res_names is not None:
        if 'res_name_list' not in metadata:
            metadata['res_name_list'] = {}
        metadata['res_name_list']['receptor'] = original_receptor_res_names
        print(f"\n alreadysavereceptorresiduetometadata({len(original_receptor_res_names)}residue)")
        print(f" PDBoutputuseresidue,notmutationN")
    
    
    
    target_coords = None  
    
    
    if (nanobody_mode or fab_mode) and cdr_regions:
        binder_slice = metadata.get('binder_slice')
        if binder_slice:
            actual_binder_length = binder_slice.stop - binder_slice.start
            if nanobody_mode:
                template_length = len(nanobody_sequence) if nanobody_sequence else 118
            elif fab_mode:
                template_length = len(fab_sequence) if fab_sequence else 440
            else:
                template_length = actual_binder_length
            
            
            if actual_binder_length != template_length:
                print(f"\n[INFO] Binderlength ({actual_binder_length}) andtemplatelength ({template_length}) not,parseCDRregion")
                if args.cdr_regions and args.cdr_regions.lower() not in ['nanobody', 'fab']:
                    cdr_regions = parse_cdr_regions(args.cdr_regions, binder_length=actual_binder_length, is_fab=fab_mode)
                    print(f" CDRregion (0-based, ):")
                    for i, (start, end) in enumerate(cdr_regions, 1):
                        print(f"    CDR{i}: [{start}, {end}) (1-based: {start+1}-{end}, {end-start}residue)")
                else:
                    
                    if nanobody_mode:
                        
                        scale = actual_binder_length / template_length
                        cdr_regions = [
                            (int(25 * scale), int(33 * scale)),
                            (int(50 * scale), int(58 * scale)),
                            (int(96 * scale), int(115 * scale)),
                        ]
                    elif fab_mode:
                        
                        scale = actual_binder_length / template_length
                        _, default_cdr_regions, _ = get_fab_template(args.fab_template)
                        cdr_regions = [(int(s * scale), int(e * scale)) for s, e in default_cdr_regions]
                    print(f" CDRregion (0-based, ):")
                    for i, (start, end) in enumerate(cdr_regions, 1):
                        print(f"    CDR{i}: [{start}, {end}) (1-based: {start+1}-{end}, {end-start}residue)")
            else:
                print(f"\n[INFO] Binderlength ({actual_binder_length}) andtemplatelength ({template_length}) match,useCDRregion")
    
    
    
    
    
    pairformer = PairformerWrapper(
        boltz_model=boltz_model,
        num_recycles=3,
        freeze_recycle_layers=True,
        monitor_z=True,
        feature_prep=feature_prep,
        use_checkpointing=False,
        use_interface_mask=False
    )
    
    diffusion = DiffusionWrapper(
        boltz_model=boltz_model,
        device=args.device
    )
    
    if args.creativity > 0.0:
        print(f"[alpha] alphafactor: {args.creativity}")
        if cdr_regions:
            print(f"[alpha] mode: Nanobody/Fab(scalingCDRZ_ij)")
        else:
            print(f"[alpha] mode: binder")
            print(f"[alpha] scaling:")
            print(f"[alpha] binder-binder: scaling")
            print(f"[alpha] binder-hotspotreceptor: scaling")
            print(f"[alpha] binder-hotspotresidue: unscaled(keep)")
            print(f"[alpha] receptor-receptor: unscaled")
    
    
    hotspot_indices = None
    if args.hotspot_indices:
        
        hotspot_indices_1based = [int(x.strip()) for x in args.hotspot_indices.split(',')]
        hotspot_indices = [idx - 1 for idx in hotspot_indices_1based]  
        print(f"Hotspot indices (1-based,input): {hotspot_indices_1based}")
        print(f"Hotspot indices (0-based,use): {hotspot_indices}")
    elif args.use_hotspot_loss:
        
        print(f"\n warning:Hotspot Lossnothotspotresidue")
        print(f" use --hotspot_indices (:--hotspot_indices '40,55,97',1-based)")
    
    
    scaling_residues = None
    if hasattr(args, '_dream_config') and args._dream_config is not None:
        config = args._dream_config
        if config.scaling_residues:
            
            receptor_slice = metadata.get('receptor_slice', slice(0, 0))
            binder_slice = metadata.get('binder_slice')
            receptor_chains = config.receptor_chains
            binder_chains = config.binder_chains
            
            scaling_residues = []
            for residue_str in config.scaling_residues:
                
                if len(residue_str) < 2:
                    print(f" warning:noscaling_residues: {residue_str},skip")
                    continue
                
                chain_id = residue_str[0].upper()
                try:
                    res_pos_1based = int(residue_str[1:])
                except ValueError:
                    print(f" warning:noparseresidueposition: {residue_str},skip")
                    continue
                
                res_pos_0based = res_pos_1based - 1  
                
                
                if chain_id in receptor_chains:
                    
                    if receptor_slice.stop > 0:
                        global_idx = receptor_slice.start + res_pos_0based
                        if receptor_slice.start <= global_idx < receptor_slice.stop:
                            scaling_residues.append(global_idx)
                            print(f" scalingresidue: {residue_str} -> index {global_idx} (receptorchain {chain_id})")
                        else:
                            print(f" warning:{residue_str} receptor [1, {receptor_slice.stop - receptor_slice.start}],skip")
                    else:
                        print(f" warning:nottoreceptorchain {chain_id},skip {residue_str}")
                elif chain_id in binder_chains:
                    
                    if binder_slice:
                        global_idx = binder_slice.start + res_pos_0based
                        if binder_slice.start <= global_idx < binder_slice.stop:
                            scaling_residues.append(global_idx)
                            print(f" scalingresidue: {residue_str} -> index {global_idx} (binderchain {chain_id})")
                        else:
                            print(f" warning:{residue_str} binder [1, {binder_slice.stop - binder_slice.start}],skip")
                    else:
                        print(f" warning:nottobinderchain {chain_id},skip {residue_str}")
                else:
                    print(f" warning:chainID {chain_id} notinreceptor_chainsorbinder_chainsin,skip {residue_str}")
            
            if scaling_residues:
                print(f"\n scalingregion: {len(scaling_residues)} residue")
                print(f" thisresidueandallresidue(ligand)Z pairscaling")
                print(f" index (0-based): {scaling_residues}")
            else:
                print(f"\n warning:scaling_residuesconfigno,nottoresidue")
                scaling_residues = None
    
    
    scaling_residues_only = False
    if hasattr(args, '_dream_config') and args._dream_config is not None:
        config = args._dream_config
        scaling_residues_only = config.scaling_residues_only
        if scaling_residues_only:
            if scaling_residues:
                print(f"\n scalingmodealready")
                print(f" scalingresidueandallresidueZ pair")
                print(f" allZ pairunscaled(defaultscaling)")
            else:
                print(f"\n warning:scaling_residues_onlyalready,scaling_residuesasempty,todefaultscaling")
                scaling_residues_only = False
    
    
    distogram_penalty_regions = None
    if args.use_distogram_penalty and args.distogram_penalty_regions:
        print(f"\n[Distogram Penalty] parseregion...")
        penalty_region_pairs = []
        for pair_str in args.distogram_penalty_regions.split(','):
            pair_str = pair_str.strip()
            if ':' not in pair_str:
                raise ValueError(f"Invalid penalty region pair format: {pair_str}. Expected format: 'region1:region2'")
            
            region1_str, region2_str = pair_str.split(':', 1)
            region1_str = region1_str.strip()
            region2_str = region2_str.strip()
            
            
            if region1_str in ['binder_framework', 'binder', 'receptor']:
                region1 = region1_str
            elif ':' in region1_str:
                
                start, end = map(int, region1_str.split(':'))
                region1 = slice(start, end)
            else:
                raise ValueError(f"Invalid region format: {region1_str}. Expected 'binder_framework', 'binder', 'receptor', or 'start:end'")
            
            
            if region2_str in ['binder_framework', 'binder', 'receptor']:
                region2 = region2_str
            elif ':' in region2_str:
                start, end = map(int, region2_str.split(':'))
                region2 = slice(start, end)
            else:
                raise ValueError(f"Invalid region format: {region2_str}. Expected 'binder_framework', 'binder', 'receptor', or 'start:end'")
            
            penalty_region_pairs.append((region1, region2))
            print(f" regionvs: {region1} × {region2}")
        
        distogram_penalty_regions = penalty_region_pairs
        print(f" {len(distogram_penalty_regions)} regionvsneed")
    elif args.use_distogram_penalty:
        
        print(f"\n[Distogram Penalty] usedefaultregion: binder:receptor(allbinderregion)")
        distogram_penalty_regions = [("binder", "receptor")]
        print(f" {len(distogram_penalty_regions)} regionvsneed")
    
    
    
    optimizable_regions = None
    if (nanobody_mode or fab_mode) and cdr_regions:
        binder_slice = metadata.get('binder_slice')
        if binder_slice:
            
            optimizable_regions = []
            for start, end in cdr_regions:
                
                
                global_start = binder_slice.start + start
                global_end = binder_slice.start + end
                optimizable_regions.append((global_start, global_end))
            print(f"\n[INFO] CDRregionalreadytosequenceindex:")
            for i, (start, end) in enumerate(optimizable_regions, 1):
                print(f"  CDR{i}: [{start}, {end}) (sequenceindex,0-based)")
        else:
            print(f"\n warning:nottobinder_slice,noCDRregiontosequenceindex")
            print(f" usevsbinderCDRregion(not)")
            optimizable_regions = cdr_regions
    
    
    if optimizable_regions is not None:
        metadata = metadata.copy() if metadata else {}
        metadata['optimizable_regions'] = optimizable_regions
    
    
    all_results = None
    
    mode1_per_round_diffusion_done = False
    generated_pdbs_mode1: List[str] = []

    
    if args.mode == 2:
        
        print("\n" + "="*80)
        print(f"mode2:mode({args.num_iterations}times)")
        print("="*80)
        print('times:')
        print(' 1. fromPairformerget(s,z)')
        print(' 2. optimization(s,z)')
        print(' 3. generatePDB(filecontains)')
        print("="*80)
        
        
        output_dir = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n output directory: {output_dir.absolute()}")
        else:
            print(f"\n output directory: currentdirectory(not--output_dir)")
        
        
        feats_for_pdb = {}
        for k, v in base_feats.items():
            if isinstance(v, torch.Tensor):
                feats_for_pdb[k] = v.cpu()
            else:
                feats_for_pdb[k] = v
        
        
        feats_for_diffusion = {}
        for k, v in base_feats.items():
            if isinstance(v, torch.Tensor):
                feats_for_diffusion[k] = v.to(args.device)
            else:
                feats_for_diffusion[k] = v
        
        for iteration in range(args.num_iterations):
            print(f"\n{'='*80}")
            print(f" {iteration + 1}/{args.num_iterations}")
            print(f"{'='*80}")
            
            
            
            if (nanobody_mode or fab_mode) and cdr_regions:
                print('\n[step0] CDR3lengthgeneratecontactvs...')
                
                
                if nanobody_mode:
                    cdr_regions = randomize_cdr3_length(
                        cdr_regions=cdr_regions,
                        binder_length=binder_length,
                        is_fab=False,  
                        random_range=3,
                        min_cdr3_length=10,  
                        max_cdr3_length=25   
                    )
                else:
                    
                    print(' ℹ Fabmode:CDRlengthkeepnot(not)')
                
                
                fixed_residue_contacts = None
                fixed_binder_positions = set()
                fixed_receptor_positions = set()
                fixed_binder_aa_map = {}
                fixed_contact_pairs = []
                
                if args.fixed_residue_contacts:
                    fixed_residue_contacts, fixed_binder_positions, fixed_receptor_positions, fixed_binder_aa_map, fixed_contact_pairs = parse_fixed_residue_contacts(
                        args.fixed_residue_contacts,
                        receptor_seq,
                        cdr_regions,
                        binder_length,
                        verbose=False  
                    )
                
                
                auto_generated_contact_pairs = None
                binder_aa_map = {}
                binder_msa_file_for_fake = None
                binder_template_modified = None
                
                if args.required_contacts:
                    receptor_positions_str = args.required_contacts.split(',')
                    receptor_indices_raw = [int(pos.strip()) - 1 for pos in receptor_positions_str if pos.strip()]
                    receptor_indices = [idx for idx in receptor_indices_raw if idx not in fixed_receptor_positions]
                    
                    if receptor_indices and cdr_regions:
                        
                        if nanobody_mode:
                            binder_seq_template = args.binder_template  
                        elif fab_mode:
                            binder_seq_template, _, _ = get_fab_template(args.fab_template)
                        else:
                            binder_seq_template = "X" * binder_length
                        
                        
                        auto_generated_contact_pairs, binder_aa_map = auto_pair_receptor_to_cdr(
                            receptor_seq=receptor_seq,
                            receptor_indices=receptor_indices,
                            cdr_regions=cdr_regions,
                            binder_length=len(binder_seq_template),
                            exclude_binder_positions=fixed_binder_positions,
                            verbose=False
                        )
                        
                        if auto_generated_contact_pairs:
                            
                            binder_seq_for_yaml = list(binder_seq_template)
                            
                            
                            if fixed_binder_aa_map:
                                for binder_idx, aa_type in fixed_binder_aa_map.items():
                                    if 0 <= binder_idx < len(binder_seq_for_yaml):
                                        binder_seq_for_yaml[binder_idx] = aa_type
                            
                            
                            for binder_idx, aa_type in binder_aa_map.items():
                                if 0 <= binder_idx < len(binder_seq_for_yaml):
                                    binder_seq_for_yaml[binder_idx] = aa_type
                            
                            binder_seq_for_yaml_str = ''.join(binder_seq_for_yaml)
                            
                            
                            binder_seq_for_msa = list(binder_seq_for_yaml_str)
                            for i in range(len(binder_seq_for_msa)):
                                if binder_seq_for_msa[i] == 'X':
                                    binder_seq_for_msa[i] = 'A'
                            binder_seq_for_msa_str = ''.join(binder_seq_for_msa)
                            
                            binder_template_modified = binder_seq_for_yaml_str
                            
                            
                            all_contact_pairs = fixed_contact_pairs.copy()
                            if auto_generated_contact_pairs:
                                all_contact_pairs.extend(auto_generated_contact_pairs)
                            
                            if all_contact_pairs:
                                try:
                                    from dream_boltz2.msa.fake_msa import generate_fake_msa_mode2
                                    
                                    
                                    temp_msa_dir = get_fake_msa_dir_from_args(args)
                                    if "fake_msa" in str(temp_msa_dir):
                                        print(f" Fake MSA saveto: {temp_msa_dir.absolute()}")
                                    receptor_df, binder_df = generate_fake_msa_mode2(
                                        receptor_seq=receptor_seq,
                                        binder_seq=binder_seq_for_msa_str,
                                        contact_pairs=all_contact_pairs,
                                        mapping_mode="complementary",
                                        num_seqs=args.max_msa_seqs,
                                        correlation_strength=0.9,
                                        output_dir=temp_msa_dir,
                                        verbose=False
                                    )
                                    
                                    receptor_msa_path = temp_msa_dir / "receptor_msa.csv"
                                    binder_msa_path = temp_msa_dir / "binder_msa.csv"
                                    receptor_df.to_csv(receptor_msa_path, index=False)
                                    binder_df.to_csv(binder_msa_path, index=False)
                                    
                                    
                                    receptor_msa_file_for_fake = str(receptor_msa_path)
                                    binder_msa_file_for_fake = str(binder_msa_path)
                                    print(f" generatefake MSA(vsMSA):")
                                    print(f"     Receptor MSA: {receptor_msa_file_for_fake}")
                                    print(f"     Binder MSA: {binder_msa_file_for_fake}")
                                except Exception as e:
                                    print(f" generatefake MSAfailed: {e}")
                
                
                if binder_msa_file_for_fake or binder_template_modified:
                    print(' (usecontactvsandfake MSA)...')
                    
                    prepare_kwargs = {
                        'receptor_sequence': receptor_seq,
                        'binder_length': binder_length,
                        'binder_cyclic': args.cyclic,
                        'receptor_msa_file': args.receptor_msa,  
                        'max_msa_seqs': args.max_msa_seqs,
                    }
                    
                    
                    if receptor_msa_file_for_fake and binder_msa_file_for_fake:
                        
                        prepare_kwargs['receptor_msa_file'] = receptor_msa_file_for_fake
                        if args.receptor_msa:
                            print(f" warning:receptor_msa,usevsfake MSA")
                        
                        prepare_kwargs['binder_msa_file'] = binder_msa_file_for_fake
                        print(f" usevsfake MSA(receptor + binder)")
                    elif binder_msa_file_for_fake:
                        
                        prepare_kwargs['binder_msa_file'] = binder_msa_file_for_fake
                        print(f" warning:binder fake MSA,vsnot")
                    
                    if fab_mode and fab_chain_info:
                        prepare_kwargs['binder_chains'] = binder_chains
                    elif binder_template_modified:
                        prepare_kwargs['binder_template'] = binder_template_modified
                    
                    if args.template_file and args.template_chain_id:
                        prepare_kwargs['template_file'] = args.template_file
                        prepare_kwargs['template_chain_id'] = args.template_chain_id
                        prepare_kwargs['template_force'] = args.template_force
                        prepare_kwargs['template_threshold'] = args.template_threshold
                    
                    base_feats, metadata = feature_prep.prepare_complex_features(**prepare_kwargs)
                    
                    
                    if cdr_regions:
                        binder_slice = metadata.get('binder_slice')
                        if binder_slice:
                            optimizable_regions = []
                            for start, end in cdr_regions:
                                
                                optimizable_regions.append(slice(binder_slice.start + start, binder_slice.start + end))
                            metadata['optimizable_regions'] = optimizable_regions
                            print(f" alreadyupdateoptimizable_regions: {len(optimizable_regions)} CDRregion")
                    
                    
                    feats_for_pdb = {}
                    for k, v in base_feats.items():
                        if isinstance(v, torch.Tensor):
                            feats_for_pdb[k] = v.cpu()
                        else:
                            feats_for_pdb[k] = v
                    
                    feats_for_diffusion = {}
                    for k, v in base_feats.items():
                        if isinstance(v, torch.Tensor):
                            feats_for_diffusion[k] = v.to(args.device)
                        else:
                            feats_for_diffusion[k] = v
            
            
            print('\n[step1] useBoltz2 forwardget(s,z)...')
            feats_with_metadata = base_feats.copy()
            feats_with_metadata['metadata'] = metadata
            
            with torch.no_grad():
                boltz_result = boltz_model.forward(
                    feats=feats_with_metadata,
                    recycling_steps=3,
                    num_sampling_steps=None,  
                    diffusion_samples=0,
                    max_parallel_samples=None,
                    run_confidence_sequentially=False,
                )
                s_new = boltz_result['s']
                z_new = boltz_result['z']
            
            print(f" get(s,z):")
            print(f"     s shape: {s_new.shape}")
            print(f"     z shape: {z_new.shape}")
            print(f"     s mean: {s_new.mean().item():.4f}, std: {s_new.std().item():.4f}")
            print(f"     z mean: {z_new.mean().item():.4f}, std: {z_new.std().item():.4f}")
            
            
            print(f"\n[step2] optimization(s,z)...")
            regeneration_attempt = 0
            while regeneration_attempt < args.max_regeneration_attempts:
                result = optimize_s_z(
                    boltz_model=boltz_model,
                    pairformer=pairformer,  
                    diffusion=diffusion,
                    feats=base_feats,
                    metadata=metadata,
                    num_steps=args.num_steps,
                    lr=args.lr,
                    device=args.device,
                    target_coords=target_coords,
                    
                    use_rg_loss=args.use_rg_loss,
                    use_helix_loss=args.use_helix_loss,
                    use_hotspot_loss=args.use_hotspot_loss,
                    rg_weight=args.rg_weight,
                    helix_weight=args.helix_weight,
                    hotspot_weight=args.hotspot_weight,
                    hotspot_indices=hotspot_indices,
                    
                    use_distogram_penalty=args.use_distogram_penalty,
                    distogram_penalty_weight=args.distogram_penalty_weight,
                    distogram_penalty_regions=distogram_penalty_regions,
                    distogram_penalty_steps=args.distogram_penalty_steps,
                    
                    optimizable_regions=optimizable_regions,
                    
                    early_filter_threshold=args.early_filter_threshold,
                    
                    creativity=args.creativity,
                    
                    scaling_residues=scaling_residues,
                    scaling_residues_only=scaling_residues_only,
                )
                
                
                if result[0] is None:
                    
                    regeneration_attempt += 1
                    print(f"\n screenfailed,generatebackbone( {regeneration_attempt}/{args.max_regeneration_attempts})...")
                    
                    
                    with torch.no_grad():
                        boltz_result = boltz_model.forward(
                            feats=feats_with_metadata,
                            recycling_steps=3,
                            num_sampling_steps=None,
                            diffusion_samples=0,
                            max_parallel_samples=None,
                            run_confidence_sequentially=False,
                        )
                        s_new = boltz_result['s']
                        z_new = boltz_result['z']
                    print(f" alreadyget(s,z)")
                    
                    
                    continue
                else:
                    
                    s_optim, z_optim, history, s_init, z_init, s_init_after_penalty, z_init_after_penalty = result
                    print(f" optimization:")
                    print(f" loss: {history['loss'][-1]:.4f}")
                    print(f" loss: {history['loss'][0]:.4f}")
                    print(f" Loss: {history['loss'][0] - history['loss'][-1]:.4f}")
                    break
            
            
            if regeneration_attempt >= args.max_regeneration_attempts:
                print(f"\n alreadytotimes ({args.max_regeneration_attempts}),skip")
                continue
            
            
            if write_coords_to_pdb is not None:
                print(f"\n[step3] generatePDB( {iteration + 1})...")
                try:
                    with torch.no_grad():
                        coords_optim = diffusion.forward(
                            s=s_optim.to(args.device),
                            z=z_optim.to(args.device),
                            feats=feats_for_diffusion,
                            num_steps=25,
                            use_fixed_noise=True,
                            start_coords=None,
                            use_physical_guidance=False,
                        )
                    
                    
                    coords_optim_cpu = coords_optim.cpu()
                    if 'token_to_center_atom' in feats_for_pdb:
                        token_to_center_atom = feats_for_pdb['token_to_center_atom']
                        if token_to_center_atom.dim() == 2:
                            token_to_center_atom = token_to_center_atom.unsqueeze(0)
                        all_ca_coords_optim = torch.bmm(token_to_center_atom.float(), coords_optim_cpu)[0]
                    elif 'token_to_rep_atom' in feats_for_pdb:
                        token_to_rep_atom = feats_for_pdb['token_to_rep_atom']
                        if token_to_rep_atom.dim() == 2:
                            token_to_rep_atom = token_to_rep_atom.unsqueeze(0)
                        all_ca_coords_optim = torch.bmm(token_to_rep_atom.float(), coords_optim_cpu)[0]
                    else:
                        raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")
                    
                    
                    if output_dir:
                        pdb_file_optim = output_dir / f"structure_optimized_sz_iter{iteration}.pdb"
                    else:
                        pdb_file_optim = Path(f"structure_optimized_sz_iter{iteration}.pdb")
                    
                    
                    metadata_for_pdb = metadata.copy()
                    
                    if config.receptor_chains and len(config.receptor_chains) > 0:
                        metadata_for_pdb['receptor_chain_id'] = config.receptor_chains[0]
                    if config.binder_chains and len(config.binder_chains) > 0:
                        metadata_for_pdb['binder_chain_id'] = config.binder_chains[0]
                    if fab_mode and fab_chain_info:
                        binder_slice = metadata['binder_slice']
                        h_chain_slice = slice(binder_slice.start, binder_slice.start + fab_chain_info['H_chain_length'])
                        l_chain_slice = slice(binder_slice.start + fab_chain_info['H_chain_length'], binder_slice.stop)
                        metadata_for_pdb['binder_chains'] = [
                            ('C', h_chain_slice),  
                            ('D', l_chain_slice)   
                        ]
                    
                    write_coords_to_pdb(
                        coords=coords_optim_cpu,
                        feats=feats_for_pdb,
                        metadata=metadata_for_pdb,
                        output_file=str(pdb_file_optim),
                        ca_coords=all_ca_coords_optim,
                        use_backbone=False,
                        use_full_sidechain=True,
                        cdr_regions=cdr_regions_for_pdb_from_metadata(metadata_for_pdb, cdr_regions) if nanobody_mode else None
                    )
                    print(f" save: {pdb_file_optim.absolute()}")
                except Exception as e:
                    print(f" generatePDBfailed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n nowrite_coords_to_pdb,skipPDBgenerate")
        
        print(f"\n{'='*80}")
        print(f" mode2！ {args.num_iterations} times")
        print(f"{'='*80}")
    else:
        
        num_design_rounds = getattr(args, 'num_design_rounds', 1)
        samples_per_round = getattr(args, 'samples_per_round', 1)
        total_structures = num_design_rounds * samples_per_round
        
        if num_design_rounds > 1 or samples_per_round > 1:
            print(f"\n roundmode")
            print(f" round: {num_design_rounds}")
            print(f" roundsampling: {samples_per_round}")
            print(f": {total_structures}")
        
        all_results = []  

        for round_idx in range(num_design_rounds):
            if num_design_rounds > 1:
                print(f"\n{'='*80}")
                print(f" roundtimes {round_idx + 1}/{num_design_rounds}")
                print(f"{'='*80}")

            
            round_feats = base_feats
            round_metadata = metadata
            round_optimizable_regions = optimizable_regions
            
            cdr_regions_binder_local = None
            if nanobody_mode and num_design_rounds > 1:
                _use_new_template = args.nanobody_template in ('random', '7eow', '7xl0', '8coh', '8z8v')
                if _use_new_template:
                    _round_seq, _round_cdr, _round_tmpl = generate_nanobody_sequence(args.nanobody_template)
                    cdr_regions_binder_local = list(_round_cdr)
                    print(f"\n roundnanobodysequence: {_round_tmpl} ({len(_round_seq)} residue)")
                    for i, (s, e) in enumerate(_round_cdr, 1):
                        print(f"    CDR{i}: [{s}, {e}) ({e-s}residue)")
                    _round_prepare_kwargs = dict(prepare_kwargs)
                    _round_prepare_kwargs['binder_template'] = _round_seq
                    _round_prepare_kwargs['binder_length'] = len(_round_seq)
                    round_feats, round_metadata = feature_prep.prepare_complex_features(**_round_prepare_kwargs)
                    
                    _binder_slice = round_metadata.get('binder_slice')
                    if _binder_slice and _round_cdr:
                        round_optimizable_regions = [
                            (_binder_slice.start + s, _binder_slice.start + e)
                            for s, e in _round_cdr
                        ]
                        round_metadata = round_metadata.copy()
                        round_metadata['optimizable_regions'] = round_optimizable_regions
                        print(f" CDRindex: {round_optimizable_regions}")

            result = optimize_s_z(
                boltz_model=boltz_model,
                pairformer=pairformer,
                diffusion=diffusion,
                feats=round_feats,
                metadata=round_metadata,
                num_steps=args.num_steps,
                lr=args.lr,
                device=args.device,
                target_coords=target_coords,
                
                use_rg_loss=args.use_rg_loss,
                use_helix_loss=args.use_helix_loss,
                use_hotspot_loss=args.use_hotspot_loss,
                rg_weight=args.rg_weight,
                helix_weight=args.helix_weight,
                hotspot_weight=args.hotspot_weight,
                hotspot_indices=hotspot_indices,
                # Distogram Penalty Loss
                use_distogram_penalty=args.use_distogram_penalty,
                distogram_penalty_weight=args.distogram_penalty_weight,
                distogram_penalty_regions=distogram_penalty_regions,
                distogram_penalty_steps=args.distogram_penalty_steps,
                
                optimizable_regions=round_optimizable_regions,
                
                early_filter_threshold=args.early_filter_threshold,
                
                creativity=args.creativity,
                
                scaling_residues=scaling_residues,
                scaling_residues_only=scaling_residues_only,
            )

            
            if result[0] is None:
                print(f"\n roundtimes {round_idx + 1} screenfailed,skip")
                continue
            
            s_optim_round, z_optim_round, history_round, s_init_round, z_init_round, s_init_ap_round, z_init_ap_round = result
            all_results.append({
                'round': round_idx + 1,
                's_optim': s_optim_round,
                'z_optim': z_optim_round,
                'history': history_round,
                'feats': round_feats,
                'metadata': round_metadata,
                'optimizable_regions': round_optimizable_regions,
                'cdr_regions_binder_local': cdr_regions_binder_local,
            })

            if write_coords_to_pdb is not None:
                _emit_mode1_round_diffusion_and_pdbs(
                    diffusion=diffusion,
                    args=args,
                    config=config,
                    nanobody_mode=nanobody_mode,
                    fab_mode=fab_mode,
                    fab_chain_info=fab_chain_info,
                    round_idx_1based=round_idx + 1,
                    num_design_rounds=num_design_rounds,
                    samples_per_round=samples_per_round,
                    s_init=s_init_round,
                    z_init=z_init_round,
                    s_optim=s_optim_round,
                    z_optim=z_optim_round,
                    s_init_after_penalty=s_init_ap_round,
                    z_init_after_penalty=z_init_ap_round,
                    round_feats=round_feats,
                    round_metadata=round_metadata,
                    cdr_regions_binder_local=cdr_regions_binder_local,
                    cdr_regions_fallback=cdr_regions,
                    generated_pdbs_accumulator=generated_pdbs_mode1,
                )
                mode1_per_round_diffusion_done = True
        
        if not all_results:
            print('\n allroundtimesfailed')
            return
        
        
        s_optim = all_results[-1]['s_optim']
        z_optim = all_results[-1]['z_optim']
        history = all_results[-1]['history']
        s_init = s_init_round if 's_init_round' in dir() else None
        z_init = z_init_round if 'z_init_round' in dir() else None
        s_init_after_penalty = None
        z_init_after_penalty = None
    
    print('\n ！')
    print(f"(s,z):")
    print(f"  s_optim: mean={s_optim.mean().item():.4f}, std={s_optim.std().item():.4f}")
    print(f"  z_optim: mean={z_optim.mean().item():.4f}, std={z_optim.std().item():.4f}")
    
    
    if write_coords_to_pdb is not None:
        if args.mode == 1 and mode1_per_round_diffusion_done:
            print("\n" + "=" * 80)
            print('Mode 1:alreadyinround GSD optimizationdiffusionand PDB(skip)')
            print("=" * 80)
            if generated_pdbs_mode1:
                print(f"\n generate {len(generated_pdbs_mode1)} PDB file")
                for _pdb_line in generated_pdbs_mode1[:50]:
                    print(f"   - {_pdb_line}")
                if len(generated_pdbs_mode1) > 50:
                    print(f"... {len(generated_pdbs_mode1) - 50} file")

        if not (args.mode == 1 and mode1_per_round_diffusion_done):
            print("\n" + "="*80)
            print('generatePDBfile(diffusionsampling)')
            print("="*80)
            
            
            output_dir = None
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n output directory: {output_dir.absolute()}")
            else:
                print(f"\n output directory: currentdirectory(not--output_dir)")
            
            
            
            
            if all_results:
                _feats_for_structure = all_results[-1]['feats']
                _metadata_for_structure = all_results[-1]['metadata']
            else:
                _feats_for_structure = base_feats
                _metadata_for_structure = metadata
            
            
            _cdr_explicit_last = all_results[-1].get('cdr_regions_binder_local') if all_results else None
            if _cdr_explicit_last:
                _cdr_for_pdb = _cdr_explicit_last
            else:
                _cdr_for_pdb = cdr_regions_for_pdb_from_metadata(_metadata_for_structure, cdr_regions)
            
            
            feats_for_pdb = {}
            for k, v in _feats_for_structure.items():
                if isinstance(v, torch.Tensor):
                    feats_for_pdb[k] = v.cpu()
                else:
                    feats_for_pdb[k] = v
            
            
            print('\n[1/2] use(s,z)diffusionsampling(sequence,optimization)...')
            print(' this(s,z)whetherstep')
            print(' if,(s,z)')
            
            
            print(f" checkfeatsinfo:")
            print(f"     feats keys: {list(_feats_for_structure.keys())[:10]}...")
            if 'res_type' in _feats_for_structure:
                print(f"     res_type shape: {_feats_for_structure['res_type'].shape}")
            if 'atom_pad_mask' in _feats_for_structure:
                print(f"     atom_pad_mask shape: {_feats_for_structure['atom_pad_mask'].shape}")
            if 'token_to_center_atom' in _feats_for_structure:
                print(f"     token_to_center_atom shape: {_feats_for_structure['token_to_center_atom'].shape}")
            
            if 'token_bonds' in _feats_for_structure:
                token_bonds = _feats_for_structure['token_bonds']
                if isinstance(token_bonds, torch.Tensor):
                    bond_count = (token_bonds > 0).sum().item()
                    print(f"     token_bonds shape: {token_bonds.shape}, {bond_count} bonds")
                else:
                    print(f"     token_bonds: {type(token_bonds)}")
            if 'connected_atom_index' in _feats_for_structure:
                connected_atom_index = _feats_for_structure['connected_atom_index']
                if isinstance(connected_atom_index, torch.Tensor):
                    if connected_atom_index.numel() > 0:
                        print(f" connected_atom_index shape: {connected_atom_index.shape}, {connected_atom_index.shape[1]} chainbondconstraint")
                    else:
                        print(f" connected_atom_index asempty(nochainbondconstraint)")
                else:
                    print(f"     connected_atom_index: {type(connected_atom_index)}")
            else:
                print(f" connected_atom_index notinfeatsin！thisbondconstraintno！")
            
            
            feats_for_diffusion = {}
            for k, v in _feats_for_structure.items():
                if isinstance(v, torch.Tensor):
                    feats_for_diffusion[k] = v.to(args.device)
                else:
                    feats_for_diffusion[k] = v
            
            
            if 'token_bonds' in feats_for_diffusion:
                token_bonds_feat = feats_for_diffusion['token_bonds']
                if isinstance(token_bonds_feat, torch.Tensor):
                    bond_count = (token_bonds_feat > 0).sum().item()
                    print(f" fordiffusionz_initcheck:")
                    print(f" token_bondsinfeatsin:, bonds={bond_count}")
                    if bond_count > 0:
                        bond_mask = (token_bonds_feat.squeeze(-1) > 0)  # [1, L, L]
                        z_init_bond_values = z_init[bond_mask.unsqueeze(-1).expand_as(z_init)]
                        print(f" z_initinbondposition: [{z_init_bond_values.min().item():.4f}, {z_init_bond_values.max().item():.4f}]")
            
            try:
                with torch.no_grad():
                    coords_init = diffusion.forward(
                        s=s_init.to(args.device),
                        z=z_init.to(args.device),
                        feats=feats_for_diffusion,  
                        num_steps=25,
                        use_fixed_noise=True,
                        start_coords=None,
                        use_physical_guidance=False,
                    )
                
                
                print(f" generatecoordinatesinfo:")
                print(f"     coords_init shape: {coords_init.shape}")
                print(f"     coords_init mean: {coords_init.mean().item():.4f}, std: {coords_init.std().item():.4f}")
                print(f"     coords_init min: {coords_init.min().item():.4f}, max: {coords_init.max().item():.4f}")
                print(f"     coords_init has NaN: {torch.isnan(coords_init).any().item()}")
                print(f"     coords_init has Inf: {torch.isinf(coords_init).any().item()}")
                
                
                coords_init_cpu = coords_init.cpu()
                if 'token_to_center_atom' in feats_for_pdb:
                    token_to_center_atom = feats_for_pdb['token_to_center_atom']
                    if token_to_center_atom.dim() == 2:
                        token_to_center_atom = token_to_center_atom.unsqueeze(0)
                    all_ca_coords_init = torch.bmm(token_to_center_atom.float(), coords_init_cpu)[0]  # [L, 3]
                    print(f" usetoken_to_center_atomextractCAcoordinates")
                elif 'token_to_rep_atom' in feats_for_pdb:
                    token_to_rep_atom = feats_for_pdb['token_to_rep_atom']
                    if token_to_rep_atom.dim() == 2:
                        token_to_rep_atom = token_to_rep_atom.unsqueeze(0)
                    all_ca_coords_init = torch.bmm(token_to_rep_atom.float(), coords_init_cpu)[0]  # [L, 3]
                    print(f" usetoken_to_rep_atomextractCAcoordinates")
                else:
                    raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")
                
                print(f" CAcoordinatesinfo:")
                print(f"     all_ca_coords_init shape: {all_ca_coords_init.shape}")
                print(f"     all_ca_coords_init mean: {all_ca_coords_init.mean().item():.4f}, std: {all_ca_coords_init.std().item():.4f}")
                print(f"     all_ca_coords_init min: {all_ca_coords_init.min().item():.4f}, max: {all_ca_coords_init.max().item():.4f}")
                print(f"     all_ca_coords_init has NaN: {torch.isnan(all_ca_coords_init).any().item()}")
                
                
                receptor_slice = _metadata_for_structure.get('receptor_slice', slice(0, _metadata_for_structure['receptor_length']))
                binder_slice = _metadata_for_structure.get('binder_slice')
                if receptor_slice and binder_slice:
                    receptor_ca = all_ca_coords_init[receptor_slice]
                    binder_ca = all_ca_coords_init[binder_slice]
                    print(f" receptorCAcoordinates (residues {receptor_slice.start}-{receptor_slice.stop-1}):")
                    print(f"     mean: {receptor_ca.mean().item():.4f}, std: {receptor_ca.std().item():.4f}")
                    print(f"     min: {receptor_ca.min().item():.4f}, max: {receptor_ca.max().item():.4f}")
                    print(f" Binder CAcoordinates (residues {binder_slice.start}-{binder_slice.stop-1}):")
                    print(f"     mean: {binder_ca.mean().item():.4f}, std: {binder_ca.std().item():.4f}")
                    print(f"     min: {binder_ca.min().item():.4f}, max: {binder_ca.max().item():.4f}")
                    
                    receptor_center = receptor_ca.mean(dim=0)
                    binder_center = binder_ca.mean(dim=0)
                    center_dist = torch.norm(receptor_center - binder_center).item()
                    print(f" receptor-binderindistance: {center_dist:.2f} Å")
                
                
                print(f" checkfeats_for_pdbsequenceinfo:")
                if 'res_type' in feats_for_pdb:
                    print(f"     res_type shape: {feats_for_pdb['res_type'].shape}")
                    print(f"     res_type dtype: {feats_for_pdb['res_type'].dtype}")
                else:
                    print(f" feats_for_pdbinnores_type！")
                
                
                
                
                if s_init_after_penalty is not None and z_init_after_penalty is not None:
                    
                    print(f" use1(s,z)(distogram)generate")
                    with torch.no_grad():
                        coords_init_after_penalty = diffusion.forward(
                            s=s_init_after_penalty.to(args.device),
                            z=z_init_after_penalty.to(args.device),
                            feats=feats_for_diffusion,
                            num_steps=25,
                            use_fixed_noise=True,
                            start_coords=None,
                                use_physical_guidance=False,
                        )
                    coords_init_after_penalty_cpu = coords_init_after_penalty.cpu()
                    if 'token_to_center_atom' in feats_for_pdb:
                        token_to_center_atom = feats_for_pdb['token_to_center_atom']
                        if token_to_center_atom.dim() == 2:
                            token_to_center_atom = token_to_center_atom.unsqueeze(0)
                        all_ca_coords_init_after_penalty = torch.bmm(token_to_center_atom.float(), coords_init_after_penalty_cpu)[0]
                    elif 'token_to_rep_atom' in feats_for_pdb:
                        token_to_rep_atom = feats_for_pdb['token_to_rep_atom']
                        if token_to_rep_atom.dim() == 2:
                            token_to_rep_atom = token_to_rep_atom.unsqueeze(0)
                        all_ca_coords_init_after_penalty = torch.bmm(token_to_rep_atom.float(), coords_init_after_penalty_cpu)[0]
                    else:
                        raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")
                    
                    
                    if output_dir:
                        pdb_file_init_after_penalty = output_dir / "structure_initial_sz_after_penalty.pdb"
                    else:
                        pdb_file_init_after_penalty = Path("structure_initial_sz_after_penalty.pdb")
                    
                    if 'token_to_center_atom' in feats_for_pdb:
                        token_to_center_atom = feats_for_pdb['token_to_center_atom']
                        if token_to_center_atom.dim() == 2:
                            token_to_center_atom = token_to_center_atom.unsqueeze(0)
                        all_ca_coords_init_after_penalty_for_pdb = torch.bmm(token_to_center_atom.float(), coords_init_after_penalty_cpu)[0]
                    elif 'token_to_rep_atom' in feats_for_pdb:
                        token_to_rep_atom = feats_for_pdb['token_to_rep_atom']
                        if token_to_rep_atom.dim() == 2:
                            token_to_rep_atom = token_to_rep_atom.unsqueeze(0)
                        all_ca_coords_init_after_penalty_for_pdb = torch.bmm(token_to_rep_atom.float(), coords_init_after_penalty_cpu)[0]
                    else:
                        raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")
                    
                    
                    metadata_for_pdb_init_after_penalty = _metadata_for_structure.copy()
                    
                    if config.receptor_chains and len(config.receptor_chains) > 0:
                        metadata_for_pdb_init_after_penalty['receptor_chain_id'] = config.receptor_chains[0]
                    if config.binder_chains and len(config.binder_chains) > 0:
                        metadata_for_pdb_init_after_penalty['binder_chain_id'] = config.binder_chains[0]
                    if fab_mode and fab_chain_info:
                        binder_slice = _metadata_for_structure['binder_slice']
                        h_chain_slice = slice(binder_slice.start, binder_slice.start + fab_chain_info['H_chain_length'])
                        l_chain_slice = slice(binder_slice.start + fab_chain_info['H_chain_length'], binder_slice.stop)
                        metadata_for_pdb_init_after_penalty['binder_chains'] = [
                            ('C', h_chain_slice),  
                            ('D', l_chain_slice)   
                        ]
                    
                    write_coords_to_pdb(
                        coords=coords_init_after_penalty_cpu,
                        feats=feats_for_pdb,
                        metadata=metadata_for_pdb_init_after_penalty,
                        output_file=str(pdb_file_init_after_penalty),
                        ca_coords=all_ca_coords_init_after_penalty_for_pdb,
                        cdr_regions=_cdr_for_pdb if nanobody_mode else None
                    )
                    print(f" save1(distogram): {pdb_file_init_after_penalty}")
                
                
                if output_dir:
                    pdb_file_init = output_dir / "structure_initial_sz_real_sequence.pdb"
                else:
                    pdb_file_init = Path("structure_initial_sz_real_sequence.pdb")
                print(f" use1(s,z)(notdistogram)generate")
                
                
                metadata_for_pdb_init = _metadata_for_structure.copy()
                
                if config.receptor_chains and len(config.receptor_chains) > 0:
                    metadata_for_pdb_init['receptor_chain_id'] = config.receptor_chains[0]
                if config.binder_chains and len(config.binder_chains) > 0:
                    metadata_for_pdb_init['binder_chain_id'] = config.binder_chains[0]
                if fab_mode and fab_chain_info:
                    binder_slice = _metadata_for_structure['binder_slice']
                    h_chain_slice = slice(binder_slice.start, binder_slice.start + fab_chain_info['H_chain_length'])
                    l_chain_slice = slice(binder_slice.start + fab_chain_info['H_chain_length'], binder_slice.stop)
                    metadata_for_pdb_init['binder_chains'] = [
                        ('C', h_chain_slice),  
                        ('D', l_chain_slice)   
                    ]
                
                write_coords_to_pdb(
                    coords=coords_init_cpu,
                    feats=feats_for_pdb,
                    metadata=metadata_for_pdb_init,
                    output_file=str(pdb_file_init),
                    ca_coords=all_ca_coords_init,
                    use_backbone=False,  
                    use_full_sidechain=True,  
                    cdr_regions=_cdr_for_pdb if nanobody_mode else None
                )
                print(f" save(sequence): {pdb_file_init}")
                print(f" ifthis,(s,z)")
            except Exception as e:
                print(f" generatePDBfailed: {e}")
                import traceback
                traceback.print_exc()
            
            
            num_design_rounds = getattr(args, 'num_design_rounds', 1)
            samples_per_round = getattr(args, 'samples_per_round', 1)
            total_structures = num_design_rounds * samples_per_round
            
            print(f"\n[2/2] useoptimization(s,z)diffusionsampling...")
            print(f" round: {num_design_rounds}")
            print(f" roundsampling: {samples_per_round}")
            print(f": {total_structures}")
            print(f" eachusenot,generate")
            
            
            metadata_for_pdb = _metadata_for_structure.copy()
            
            if config.receptor_chains and len(config.receptor_chains) > 0:
                metadata_for_pdb['receptor_chain_id'] = config.receptor_chains[0]
            if config.binder_chains and len(config.binder_chains) > 0:
                metadata_for_pdb['binder_chain_id'] = config.binder_chains[0]
            if fab_mode and fab_chain_info:
                binder_slice = _metadata_for_structure['binder_slice']
                h_chain_slice = slice(binder_slice.start, binder_slice.start + fab_chain_info['H_chain_length'])
                l_chain_slice = slice(binder_slice.start + fab_chain_info['H_chain_length'], binder_slice.stop)
                metadata_for_pdb['binder_chains'] = [
                    ('C', h_chain_slice),  
                    ('D', l_chain_slice)   
                ]
            
            generated_pdbs = []
            
            
            if 'token_bonds' in feats_for_diffusion and 's_optim' in locals() and 'z_optim' in locals():
                token_bonds_feat = feats_for_diffusion['token_bonds']
                if isinstance(token_bonds_feat, torch.Tensor):
                    bond_count = (token_bonds_feat > 0).sum().item()
                    bond_mask = (token_bonds_feat.squeeze(-1) > 0)  # [1, L, L]
                    z_optim_bond_values = z_optim[bond_mask.unsqueeze(-1).expand_as(z_optim)]
                    z_init_bond_values = z_init[bond_mask.unsqueeze(-1).expand_as(z_init)]
                    if z_optim_bond_values.numel() > 0 and z_init_bond_values.numel() > 0:
                        mean_diff = (z_optim_bond_values.mean() - z_init_bond_values.mean()).abs().item()
                        max_diff = (z_optim_bond_values - z_init_bond_values).abs().max().item()
                        print(f"\n [Z_OPTIM DEBUG] checkoptimizationz_optimwhethertoken_bondsinfo:")
                        print(f" z_optiminbondposition: [{z_optim_bond_values.min().item():.4f}, {z_optim_bond_values.max().item():.4f}]")
                        print(f" z_initinbondposition: [{z_init_bond_values.min().item():.4f}, {z_init_bond_values.max().item():.4f}]")
                        print(f": {mean_diff:.4f} (if<1.0,token_bondsinfo)")
                        print(f": {max_diff:.4f} (if<1.0,token_bondsinfo)")
                    else:
                        print(f"\n [Z_OPTIM DEBUG] nottobondposition,skipz_optimcheck")
            
            # ------------------------------------------------------------------
            
            
            
            # ------------------------------------------------------------------
            if False:
                
                for round_result in all_results:
                    round_idx = round_result['round']
                    s_optim_round = round_result['s_optim']
                    z_optim_round = round_result['z_optim']
    
                    
                    _nanobody_random = nanobody_mode and args.nanobody_template in ('random', '7eow', '7xl0', '8coh', '8z8v')
                    if _nanobody_random:
                        _round_feats = round_result['feats']
                        _round_metadata = round_result['metadata']
                        feats_for_diffusion_round = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in _round_feats.items()}
                        feats_for_pdb_round = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in _round_feats.items()}
                        metadata_for_pdb_round = _round_metadata.copy()
                        if config.receptor_chains and len(config.receptor_chains) > 0:
                            metadata_for_pdb_round['receptor_chain_id'] = config.receptor_chains[0]
                        if config.binder_chains and len(config.binder_chains) > 0:
                            metadata_for_pdb_round['binder_chain_id'] = config.binder_chains[0]
                    else:
                        feats_for_diffusion_round = feats_for_diffusion
                        feats_for_pdb_round = feats_for_pdb
                        metadata_for_pdb_round = metadata_for_pdb
    
                    print(f"\n roundtimes {round_idx}/{num_design_rounds}:")
                    
                    for sample_idx in range(samples_per_round):
                        try:
                            
                            if sample_idx == 0 and 'token_bonds' in feats_for_diffusion_round:
                                token_bonds_feat = feats_for_diffusion_round['token_bonds']
                                if isinstance(token_bonds_feat, torch.Tensor):
                                    bond_count = (token_bonds_feat > 0).sum().item()
                                    bond_mask = (token_bonds_feat.squeeze(-1) > 0)  # [1, L, L]
                                    z_optim_bond_values = z_optim_round[bond_mask.unsqueeze(-1).expand_as(z_optim_round)]
                                    if z_optim_bond_values.numel() > 0:
                                        print(f" [Z_OPTIM DEBUG] checkoptimizationz_optimwhethertoken_bondsinfo:")
                                        print(f" z_optiminbondposition: [{z_optim_bond_values.min().item():.4f}, {z_optim_bond_values.max().item():.4f}]")
                                    else:
                                        print(f" [Z_OPTIM DEBUG] nottobondposition,skipz_optimcheck")
    
                            with torch.no_grad():
                                coords_optim = diffusion.forward(
                                    s=s_optim_round.to(args.device),
                                    z=z_optim_round.to(args.device),
                                    feats=feats_for_diffusion_round,
                                    num_steps=25,
                                    use_fixed_noise=False,  
                                    start_coords=None,
                                    use_physical_guidance=False,
                                )
    
                            coords_optim_cpu = coords_optim.cpu()
                            if 'token_to_center_atom' in feats_for_pdb_round:
                                token_to_center_atom = feats_for_pdb_round['token_to_center_atom']
                                if token_to_center_atom.dim() == 2:
                                    token_to_center_atom = token_to_center_atom.unsqueeze(0)
                                all_ca_coords_optim = torch.bmm(token_to_center_atom.float(), coords_optim_cpu)[0]
                            elif 'token_to_rep_atom' in feats_for_pdb_round:
                                token_to_rep_atom = feats_for_pdb_round['token_to_rep_atom']
                                if token_to_rep_atom.dim() == 2:
                                    token_to_rep_atom = token_to_rep_atom.unsqueeze(0)
                                all_ca_coords_optim = torch.bmm(token_to_rep_atom.float(), coords_optim_cpu)[0]
                            else:
                                raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")
    
                            
                            if sample_idx == 0 and 'connected_atom_index' in feats_for_diffusion_round:
                                connected_atom_index = feats_for_diffusion_round['connected_atom_index']
                                if isinstance(connected_atom_index, torch.Tensor) and connected_atom_index.numel() > 0:
                                    
                                    if connected_atom_index.dim() == 3:
                                        connected_atom_index = connected_atom_index.squeeze(0)  # [2, N]
                                    if connected_atom_index.shape[0] == 2 and connected_atom_index.shape[1] > 0:
                                        atom1_idx = connected_atom_index[0, 0].item()
                                        atom2_idx = connected_atom_index[1, 0].item()
                                        if atom1_idx < coords_optim_cpu.shape[1] and atom2_idx < coords_optim_cpu.shape[1]:
                                            atom1_coord = coords_optim_cpu[0, atom1_idx, :]  # [3]
                                            atom2_coord = coords_optim_cpu[0, atom2_idx, :]  # [3]
                                            bond_distance = torch.norm(atom1_coord - atom2_coord).item()
                                            print(f" [BOND CONSTRAINT CHECK] {sample_idx + 1}:")
                                            print(f" Bondconstraintatomvs: [{atom1_idx}, {atom2_idx}]")
                                            print(f" Bonddistance: {bond_distance:.2f} Å (: ~1.5-2.0 Å for covalent bond)")
                                            if bond_distance > 3.0:
                                                print(f" Bonddistance！constraintnot！")
                                            else:
                                                print(f" Bonddistance,constraint")
    
                            
                            if output_dir:
                                pdb_file_optim = output_dir / f"structure_round{round_idx:02d}_sample{sample_idx + 1:02d}.pdb"
                            else:
                                pdb_file_optim = Path(f"structure_round{round_idx:02d}_sample{sample_idx + 1:02d}.pdb")
    
                            _cdr_explicit_r = round_result.get('cdr_regions_binder_local')
                            if _cdr_explicit_r:
                                _cdr_round_pdb = _cdr_explicit_r
                            else:
                                _cdr_round_pdb = cdr_regions_for_pdb_from_metadata(round_result.get('metadata'), cdr_regions)
                            write_coords_to_pdb(
                                coords=coords_optim_cpu,
                                feats=feats_for_pdb_round,
                                metadata=metadata_for_pdb_round,
                                output_file=str(pdb_file_optim),
                                ca_coords=all_ca_coords_optim,
                                use_backbone=False,
                                use_full_sidechain=True,
                                cdr_regions=_cdr_round_pdb if nanobody_mode else None
                            )
                            generated_pdbs.append(str(pdb_file_optim))
                            print(f" {sample_idx + 1}/{samples_per_round}: {pdb_file_optim}")
                        except Exception as e:
                            print(f" {sample_idx + 1} generatefailed: {e}")
            else:
                
                
                if 'token_bonds' in feats_for_diffusion and 's_optim' in locals() and 'z_optim' in locals():
                    token_bonds_feat = feats_for_diffusion['token_bonds']
                    if isinstance(token_bonds_feat, torch.Tensor):
                        bond_count = (token_bonds_feat > 0).sum().item()
                        bond_mask = (token_bonds_feat.squeeze(-1) > 0)  # [1, L, L]
                        z_optim_bond_values = z_optim[bond_mask.unsqueeze(-1).expand_as(z_optim)]
                        z_init_bond_values = z_init[bond_mask.unsqueeze(-1).expand_as(z_init)]
                        if z_optim_bond_values.numel() > 0 and z_init_bond_values.numel() > 0:
                            mean_diff = (z_optim_bond_values.mean() - z_init_bond_values.mean()).abs().item()
                            max_diff = (z_optim_bond_values - z_init_bond_values).abs().max().item()
                            print(f"\n [Z_OPTIM DEBUG] checkoptimizationz_optimwhethertoken_bondsinfo:")
                            print(f" z_optiminbondposition: [{z_optim_bond_values.min().item():.4f}, {z_optim_bond_values.max().item():.4f}]")
                            print(f" z_initinbondposition: [{z_init_bond_values.min().item():.4f}, {z_init_bond_values.max().item():.4f}]")
                            print(f": {mean_diff:.4f} (if<1.0,token_bondsinfo)")
                            print(f": {max_diff:.4f} (if<1.0,token_bondsinfo)")
                        else:
                            print(f"\n [Z_OPTIM DEBUG] nottobondposition,skipz_optimcheck")
                
                for sample_idx in range(samples_per_round):
                    try:
                        with torch.no_grad():
                            coords_optim = diffusion.forward(
                                s=s_optim.to(args.device),
                                z=z_optim.to(args.device),
                                feats=feats_for_diffusion,
                                num_steps=25,
                                use_fixed_noise=False,  
                                start_coords=None,
                                use_physical_guidance=False,
                            )
                        
                        coords_optim_cpu = coords_optim.cpu()
                        if 'token_to_center_atom' in feats_for_pdb:
                            token_to_center_atom = feats_for_pdb['token_to_center_atom']
                            if token_to_center_atom.dim() == 2:
                                token_to_center_atom = token_to_center_atom.unsqueeze(0)
                            all_ca_coords_optim = torch.bmm(token_to_center_atom.float(), coords_optim_cpu)[0]
                        elif 'token_to_rep_atom' in feats_for_pdb:
                            token_to_rep_atom = feats_for_pdb['token_to_rep_atom']
                            if token_to_rep_atom.dim() == 2:
                                token_to_rep_atom = token_to_rep_atom.unsqueeze(0)
                            all_ca_coords_optim = torch.bmm(token_to_rep_atom.float(), coords_optim_cpu)[0]
                        else:
                            raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")
                        
                        
                        if sample_idx == 0 and 'connected_atom_index' in feats_for_diffusion:
                            connected_atom_index = feats_for_diffusion['connected_atom_index']
                            if isinstance(connected_atom_index, torch.Tensor) and connected_atom_index.numel() > 0:
                                
                                if connected_atom_index.dim() == 3:
                                    connected_atom_index = connected_atom_index.squeeze(0)  # [2, N]
                                if connected_atom_index.shape[0] == 2 and connected_atom_index.shape[1] > 0:
                                    atom1_idx = connected_atom_index[0, 0].item()
                                    atom2_idx = connected_atom_index[1, 0].item()
                                    if atom1_idx < coords_optim_cpu.shape[1] and atom2_idx < coords_optim_cpu.shape[1]:
                                        atom1_coord = coords_optim_cpu[0, atom1_idx, :]  # [3]
                                        atom2_coord = coords_optim_cpu[0, atom2_idx, :]  # [3]
                                        bond_distance = torch.norm(atom1_coord - atom2_coord).item()
                                        print(f" [BOND CONSTRAINT CHECK] {sample_idx + 1}:")
                                        print(f" Bondconstraintatomvs: [{atom1_idx}, {atom2_idx}]")
                                        print(f" Bonddistance: {bond_distance:.2f} Å (: ~1.5-2.0 Å for covalent bond)")
                                        if bond_distance > 3.0:
                                            print(f" Bonddistance！constraintnot！")
                                        else:
                                            print(f" Bonddistance,constraint")
                        
                        
                        if samples_per_round == 1:
                            
                            if output_dir:
                                pdb_file_optim = output_dir / "structure_optimized_sz.pdb"
                            else:
                                pdb_file_optim = Path("structure_optimized_sz.pdb")
                        else:
                            
                            if output_dir:
                                pdb_file_optim = output_dir / f"structure_sample{sample_idx + 1:02d}.pdb"
                            else:
                                pdb_file_optim = Path(f"structure_sample{sample_idx + 1:02d}.pdb")
                        
                        write_coords_to_pdb(
                            coords=coords_optim_cpu,
                            feats=feats_for_pdb,
                            metadata=metadata_for_pdb,
                            output_file=str(pdb_file_optim),
                            ca_coords=all_ca_coords_optim,
                            use_backbone=False,
                            use_full_sidechain=True,
                            cdr_regions=_cdr_for_pdb if nanobody_mode else None
                        )
                        generated_pdbs.append(str(pdb_file_optim))
                        print(f" {sample_idx + 1}/{samples_per_round}: {pdb_file_optim}")
                    except Exception as e:
                        print(f" {sample_idx + 1} generatefailed: {e}")
                        import traceback
                        traceback.print_exc()
            
            print(f"\n PDBfilegenerate！generate {len(generated_pdbs)} ")
            if len(generated_pdbs) > 0:
                print(f" generate:")
                for pdb_path in generated_pdbs:
                    print(f"   - {pdb_path}")
    else:
        print('\n nowrite_coords_to_pdb,skipPDBgenerate')


def main():
    """main."""
    parser = _build_optimize_arg_parser()
    args = parser.parse_args()
    _run_with_args(args, parser=parser)


if __name__ == "__main__":
    main()

