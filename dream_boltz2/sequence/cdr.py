"""Optional CDR B-factor annotation for antibody inputs."""

import random
import os
from typing import List, Tuple, Optional, Dict


# ============================================================================
# 4-Template Nanobody System (aligned with boltzgen)
# ============================================================================

NANOBODY_TEMPLATES = {
    '7eow': {
        'fr1': 'EVQLVESGGGLVQPGGSLRLSCAAS',       # 25
        'cdr1_range': (7, 11),
        'fr2': 'GWFRQAPGKGRELVAAI',                # 17
        'cdr2_range': (6, 10),
        'fr3': 'YPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCA',  # 38
        'cdr3_range': (15, 28),
        'fr4': 'GQGTQVTVSS',                       # 10
    },
    '7xl0': {
        'fr1': 'EVQLVESGGGLVQPGGSLRLSCAAS',        # 25
        'cdr1_range': (6, 10),
        'fr2': 'MAWYRQAPGKGRELVAG',                # 17
        'cdr2_range': (5, 9),
        'fr3': 'SYADSVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCA',  # 39
        'cdr3_range': (9, 20),
        'fr4': 'WGQGTLVTVSS',                      # 11
    },
    '8coh': {
        'fr1': 'EVQLVESGGGLVQPGGSLRLSCAAS',        # 25
        'cdr1_range': (6, 10),
        'fr2': 'MAWFRQAPGQEREFVAG',                # 17
        'cdr2_range': (6, 10),
        'fr3': 'LYADSVRGRFTNSRDNSKNTLYLQMNSLRAEDTAVYYC',  # 38
        'cdr3_range': (13, 26),
        'fr4': 'WGQGTLVTVSS',                      # 11
    },
    '8z8v': {
        'fr1': 'EVQLVESGGGLVQPGNSLRLSCAAS',        # 25
        'cdr1_range': (6, 10),
        'fr2': 'MSWVRQAPGKGLEWVSS',                # 17
        'cdr2_range': (6, 10),
        'fr3': 'LYADSVKGRFTISRDNAKTTLYLQMNSLRPEDTAVYYCT',  # 39
        'cdr3_range': (9, 20),
        'fr4': 'TLVTVSS',                           # 7
    },
}

# CDR filling weights: Y(30%) F(20%) R(20%) D(10%) W(20%)
CDR_FILL_WEIGHTS = {'Y': 0.30, 'F': 0.20, 'R': 0.20, 'D': 0.10, 'W': 0.20}

_CDR_FILL_AAS = list(CDR_FILL_WEIGHTS.keys())
_CDR_FILL_PROBS = list(CDR_FILL_WEIGHTS.values())


def fill_cdr_weighted_random(length: int) -> str:
    """Fill a CDR region with weighted random amino acids (Y/F/R/D/W)."""
    return ''.join(random.choices(_CDR_FILL_AAS, weights=_CDR_FILL_PROBS, k=length))


def generate_nanobody_sequence(
    template_name: str = 'random',
) -> Tuple[str, List[Tuple[int, int]], str]:
    """generate nanobody sequence."""
    if template_name == 'random':
        template_name_used = random.choice(list(NANOBODY_TEMPLATES.keys()))
    elif template_name in NANOBODY_TEMPLATES:
        template_name_used = template_name
    else:
        raise ValueError(
            f"Unknown nanobody template: {template_name}. "
            f"Options: random, {', '.join(NANOBODY_TEMPLATES.keys())}"
        )

    tmpl = NANOBODY_TEMPLATES[template_name_used]

    # Sample random CDR lengths within range (inclusive)
    cdr1_len = random.randint(*tmpl['cdr1_range'])
    cdr2_len = random.randint(*tmpl['cdr2_range'])
    cdr3_len = random.randint(*tmpl['cdr3_range'])

    # Fill CDR with weighted random amino acids
    cdr1 = fill_cdr_weighted_random(cdr1_len)
    cdr2 = fill_cdr_weighted_random(cdr2_len)
    cdr3 = fill_cdr_weighted_random(cdr3_len)

    # Build sequence
    fr1 = tmpl['fr1']
    fr2 = tmpl['fr2']
    fr3 = tmpl['fr3']
    fr4 = tmpl['fr4']

    sequence = fr1 + cdr1 + fr2 + cdr2 + fr3 + cdr3 + fr4

    # Compute CDR regions (0-based, half-open)
    cdr1_start = len(fr1)
    cdr1_end = cdr1_start + cdr1_len

    cdr2_start = cdr1_end + len(fr2)
    cdr2_end = cdr2_start + cdr2_len

    cdr3_start = cdr2_end + len(fr3)
    cdr3_end = cdr3_start + cdr3_len

    cdr_regions = [
        (cdr1_start, cdr1_end),
        (cdr2_start, cdr2_end),
        (cdr3_start, cdr3_end),
    ]

    return sequence, cdr_regions, template_name_used


def detect_cdr_regions_from_bfactor(
    pdb_path: str,
    binder_chain: str,
    threshold: float = 0.99,
) -> List[Tuple[int, int]]:
    """detect cdr regions from bfactor."""
    # Parse residue B-factors from PDB (use CA atoms)
    residue_bfactors: Dict[int, float] = {}  # res_num (1-based) -> bfactor

    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                chain = line[21:22].strip()
                if chain != binder_chain:
                    continue
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                res_num = int(line[22:26].strip())
                try:
                    bfactor = float(line[60:66].strip())
                except (ValueError, IndexError):
                    continue
                residue_bfactors[res_num] = bfactor
    except Exception as e:
        print(f"  [detect_cdr_regions_from_bfactor] Failed to read PDB: {e}")
        return []

    if not residue_bfactors:
        return []

    # Sort by residue number
    sorted_res = sorted(residue_bfactors.keys())

    # Find consecutive runs of low B-factor residues (CDR)
    cdr_regions = []
    run_start = None

    for res_num in sorted_res:
        is_cdr = residue_bfactors[res_num] < threshold
        if is_cdr:
            if run_start is None:
                run_start = res_num
        else:
            if run_start is not None:
                # Convert 1-based res_num to 0-based index relative to binder
                start_0based = sorted_res.index(run_start)
                end_0based = sorted_res.index(res_num)  # exclusive
                cdr_regions.append((start_0based, end_0based))
                run_start = None

    # Handle run extending to end
    if run_start is not None:
        start_0based = sorted_res.index(run_start)
        end_0based = len(sorted_res)
        cdr_regions.append((start_0based, end_0based))

    return cdr_regions


# ============================================================================
# Sequence-based CDR detection & B-factor validation/repair
# ============================================================================


_AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def extract_sequence_from_pdb(
    pdb_path: str,
    chain_id: str,
) -> str:
    """extract sequence from pdb."""
    residues = {}  # res_num -> res_name_3letter

    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            chain = line[21:22].strip()
            if chain != chain_id:
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            res_num = int(line[22:26].strip())
            res_name = line[17:20].strip()
            residues[res_num] = res_name

    
    sequence = ''
    for res_num in sorted(residues.keys()):
        aa = _AA_3TO1.get(residues[res_num], 'X')
        sequence += aa

    return sequence


def detect_cdr_regions_from_sequence(
    sequence: str,
) -> Tuple[Optional[str], List[Tuple[int, int]]]:
    """detect cdr regions from sequence."""
    best_template = None
    best_cdr_regions = []
    best_score = -1.0

    for tmpl_name, tmpl in NANOBODY_TEMPLATES.items():
        fr1 = tmpl['fr1']
        fr2 = tmpl['fr2']
        fr3 = tmpl['fr3']
        fr4 = tmpl['fr4']

        
        
        fr1_pos = _find_best_match(sequence, fr1, start=0, end=len(fr1) + 10)
        if fr1_pos < 0:
            continue
        fr1_end = fr1_pos + len(fr1)  

        
        search_start = fr1_end + tmpl['cdr1_range'][0] - 2  
        search_end = fr1_end + tmpl['cdr1_range'][1] + 5    
        fr2_pos = _find_best_match(sequence, fr2, start=search_start, end=min(search_end + len(fr2), len(sequence)))
        if fr2_pos < 0:
            continue
        fr2_end = fr2_pos + len(fr2)  

        
        search_start = fr2_end + tmpl['cdr2_range'][0] - 2
        search_end = fr2_end + tmpl['cdr2_range'][1] + 5
        fr3_pos = _find_best_match(sequence, fr3, start=search_start, end=min(search_end + len(fr3), len(sequence)))
        if fr3_pos < 0:
            continue
        fr3_end = fr3_pos + len(fr3)  

        
        search_start = fr3_end + tmpl['cdr3_range'][0] - 2
        search_end = fr3_end + tmpl['cdr3_range'][1] + 5
        fr4_pos = _find_best_match(sequence, fr4, start=search_start, end=min(search_end + len(fr4), len(sequence)))
        if fr4_pos < 0:
            continue

        
        cdr1 = (fr1_end, fr2_pos)
        cdr2 = (fr2_end, fr3_pos)
        cdr3 = (fr3_end, fr4_pos)

        
        if cdr1[1] <= cdr1[0] or cdr2[1] <= cdr2[0] or cdr3[1] <= cdr3[0]:
            continue

        
        
        score = 0.0
        for fr_seq, fr_pos in [(fr1, fr1_pos), (fr2, fr2_pos), (fr3, fr3_pos), (fr4, fr4_pos)]:
            subseq = sequence[fr_pos:fr_pos + len(fr_seq)]
            score += sum(1 for a, b in zip(subseq, fr_seq) if a == b)

        if score > best_score:
            best_score = score
            best_template = tmpl_name
            best_cdr_regions = [cdr1, cdr2, cdr3]

    return best_template, best_cdr_regions


def _find_best_match(
    sequence: str,
    pattern: str,
    start: int = 0,
    end: Optional[int] = None,
    min_identity: float = 0.75,
) -> int:
    """ find best match."""
    start = max(0, start)
    if end is None:
        end = len(sequence)
    end = min(end, len(sequence))
    pat_len = len(pattern)

    if end - start < pat_len:
        
        start = max(0, start - 5)
        end = min(len(sequence), end + pat_len + 5)

    
    idx = sequence.find(pattern, start, end)
    if idx >= 0:
        return idx

    
    best_pos = -1
    best_identity = 0.0

    for pos in range(start, min(end, len(sequence) - pat_len + 1)):
        subseq = sequence[pos:pos + pat_len]
        matches = sum(1 for a, b in zip(subseq, pattern) if a == b)
        identity = matches / pat_len
        if identity > best_identity:
            best_identity = identity
            best_pos = pos

    if best_identity >= min_identity:
        return best_pos

    return -1


def validate_and_fix_cdr_bfactor(
    pdb_path: str,
    binder_chain: str,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[bool, List[Tuple[int, int]], Optional[str]]:
    """validate and fix cdr bfactor."""
    if output_path is None:
        output_path = pdb_path

    
    bfactor_cdr = detect_cdr_regions_from_bfactor(pdb_path, binder_chain)

    
    sequence = extract_sequence_from_pdb(pdb_path, binder_chain)
    if not sequence:
        if verbose:
            print(f"  [validate_cdr] WARNING: Could not extract sequence from chain {binder_chain}")
        return True, bfactor_cdr, None  

    matched_template, seq_cdr = detect_cdr_regions_from_sequence(sequence)

    if matched_template is None:
        if verbose:
            print(f"  [validate_cdr] WARNING: No template matched for sequence (len={len(sequence)})")
            print(f"  [validate_cdr] Sequence: {sequence[:30]}...{sequence[-15:]}")
            print(f"  [validate_cdr] Falling back to B-factor CDR regions")
        return True, bfactor_cdr, None  

    if verbose:
        print(f"  [validate_cdr] Matched template: {matched_template}")
        print(f"  [validate_cdr] Sequence length: {len(sequence)}")
        print(f"  [validate_cdr] Sequence CDR regions: {seq_cdr}")
        print(f"  [validate_cdr] B-factor CDR regions: {bfactor_cdr}")

    
    is_consistent = (bfactor_cdr == seq_cdr)

    if is_consistent:
        if verbose:
            print(f"  [validate_cdr] OK - B-factor and sequence CDR regions are CONSISTENT")
        return True, seq_cdr, matched_template

    
    if verbose:
        print(f"  [validate_cdr] MISMATCH detected! Rewriting B-factor based on sequence alignment")
        
        if len(bfactor_cdr) != len(seq_cdr):
            print(f"  [validate_cdr]   CDR count differs: B-factor={len(bfactor_cdr)}, sequence={len(seq_cdr)}")
        else:
            for i, (bf_region, seq_region) in enumerate(zip(bfactor_cdr, seq_cdr)):
                if bf_region != seq_region:
                    print(f"  [validate_cdr]   CDR{i+1}: B-factor={bf_region}, sequence={seq_region}")

    _rewrite_bfactor_in_pdb(pdb_path, binder_chain, seq_cdr, output_path)

    if verbose:
        print(f"  [validate_cdr] OK - B-factor repaired -> {output_path}")

    return False, seq_cdr, matched_template


def _rewrite_bfactor_in_pdb(
    pdb_path: str,
    binder_chain: str,
    cdr_regions: List[Tuple[int, int]],
    output_path: str,
):
    """ rewrite bfactor in pdb."""
    
    
    res_num_to_index = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            chain = line[21:22].strip()
            if chain != binder_chain:
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            res_num = int(line[22:26].strip())
            if res_num not in res_num_to_index:
                res_num_to_index[res_num] = len(res_num_to_index)

    def _is_cdr(local_idx: int) -> bool:
        for cdr_start, cdr_end in cdr_regions:
            if cdr_start <= local_idx < cdr_end:
                return True
        return False

    
    lines_out = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21:22].strip()
                if chain == binder_chain:
                    res_num = int(line[22:26].strip())
                    local_idx = res_num_to_index.get(res_num)
                    if local_idx is not None:
                        bfac = 0.0 if _is_cdr(local_idx) else 100.0
                        # PDB format: columns 61-66 = B-factor (6.2f)
                        line = line[:60] + f"{bfac:6.2f}" + line[66:]
                lines_out.append(line)
            else:
                lines_out.append(line)

    with open(output_path, 'w') as f:
        f.writelines(lines_out)


# ============================================================================
# Backward-compatible API
# ============================================================================

def get_nanobody_template(template_name: str = 'nanobody_default') -> Tuple[str, List[Tuple[int, int]]]:
    """get nanobody template."""
    if template_name == 'nanobody_default':
        
        fr1 = "EVQLVESGGGLVQPGGSLRLSCAAS"  # 25
        cdr1 = "X" * 8
        fr2 = "MAWFRQAPGQEREFVAG"          # 17
        cdr2 = "X" * 8
        fr3 = "LYADSVRGRFTNSRDNSKNTLYLQMNSLRAEDTAVYYC"  # 38
        cdr3 = "X" * 19
        fr4 = "WGQGTLVTVSS"                # 11

        sequence = fr1 + cdr1 + fr2 + cdr2 + fr3 + cdr3 + fr4

        cdr_regions = [
            (25, 33),   # CDR1
            (50, 58),   # CDR2
            (96, 115),  # CDR3
        ]

        return sequence, cdr_regions
    else:
        # Delegate to generate_nanobody_sequence for new template names
        sequence, cdr_regions, _ = generate_nanobody_sequence(template_name)
        return sequence, cdr_regions


def get_fab_template(template_name: str = 'fab_default') -> Tuple[str, List[Tuple[int, int]], Dict]:
    """get fab template."""
    if template_name == 'fab_default':
        h_chain = "EVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXWVRQAPGKGLEWVAXXXXXXXXXXXXXXXXXRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRXXXXXXXXXXXWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKS"
        l_chain = "DIQMTQSPSSLSASVGDRVTITCXXXXXXXXXXXWYQQKPGKAPKLLIYXXXXXXXGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCXXXXXXXXXFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGES"

        h_chain_length = len(h_chain)
        l_chain_length = len(l_chain)
        sequence = h_chain + l_chain
        total_length = len(sequence)

        
        def find_cdr_regions(seq: str, offset: int = 0) -> List[Tuple[int, int]]:
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

        return sequence, cdr_regions, chain_info
    else:
        raise ValueError(f"notFabtemplate: {template_name}")


def parse_cdr_regions(cdr_str: str, binder_length: int, is_fab: bool = False) -> List[Tuple[int, int]]:
    """parse cdr regions."""
    if cdr_str.lower() == 'fab':
        
        if is_fab:
            _, cdr_regions, _ = get_fab_template('fab_default')
            return cdr_regions
        else:
            raise ValueError("Fab CDR regions require is_fab=True")
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
