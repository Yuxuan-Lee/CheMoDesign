#!/usr/bin/env python3
"""Write AlphaFold 3 JSON inputs from designed complexes."""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List


def three_to_one_letter(three_letter_code: str) -> str:
    """three to one letter."""
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEC': 'U', 'PYL': 'O',  
        'UNK': 'X', 'MSE': 'M',  
    }
    return aa_dict.get(three_letter_code.upper(), 'X')


def extract_sequence_from_cif(cif_file: Path, chain_id: str) -> Tuple[Optional[str], Optional[str]]:
    """extract sequence from cif."""
    try:
        sequence_dict = {}  # {seq_id: aa}
        seen_residues = set()  
        in_atom_site = False
        atom_site_cols = {}  
        
        with open(cif_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        
        in_loop = False
        
        
        has_atom_lines = any(l.strip().startswith('ATOM') or l.strip().startswith('HETATM') for l in lines)
        
        
        if not has_atom_lines:
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if line_stripped == 'loop_':
                    in_loop = True
                elif in_loop and line_stripped.startswith('_atom_site.'):
                    
                    col_name = line_stripped.replace('_atom_site.', '').strip()
                    atom_site_cols[col_name] = len(atom_site_cols)
                elif in_loop and atom_site_cols and not line_stripped.startswith('_atom_site.'):
                    
                    if line_stripped and not line_stripped.startswith('#'):
                        
                        break
        
        
        
        use_fixed_format = has_atom_lines and len(atom_site_cols) == 0
        
        
        if not has_atom_lines and len(atom_site_cols) == 0:
            return None, 'nottoATOMoratom_site'
        
        if not use_fixed_format:
            
            required_cols = ['label_comp_id', 'label_asym_id', 'label_seq_id']
            missing_cols = [col for col in required_cols if col not in atom_site_cols]
            if missing_cols:
                return None, f": {missing_cols}"
        
        
        if use_fixed_format:
            
            atom_count = 0
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith('ATOM') or line_stripped.startswith('HETATM'):
                    parts = line.split()
                    
                    
                    if len(parts) >= 9:
                        try:
                            comp_id = parts[5].strip()  
                            chain = parts[6].strip()    
                            seq_id_str = parts[8].strip()  
                            atom_name = parts[3].strip() if len(parts) > 3 else ''
                            is_ca = (atom_name == 'CA')
                            
                            if chain == chain_id and is_ca:
                                try:
                                    seq_id = int(seq_id_str)
                                    residue_key = f"{chain}_{seq_id}"
                                    if residue_key not in seen_residues:
                                        seen_residues.add(residue_key)
                                        aa_single = three_to_one_letter(comp_id)
                                        if aa_single and aa_single != 'X':
                                            sequence_dict[seq_id] = aa_single
                                            atom_count += 1
                                except (ValueError, IndexError):
                                    continue
                        except (IndexError, ValueError):
                            continue
        else:
            
            max_col_idx = max(atom_site_cols.values()) if atom_site_cols else -1
            data_started = False
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                
                
                if data_started and (line_stripped == 'loop_' or 
                                     (line_stripped.startswith('_') and not line_stripped.startswith('_atom_site.'))):
                    break
                
                
                if not line_stripped.startswith('_') and not line_stripped.startswith('loop_'):
                    parts = line.split()
                    
                    
                    if len(parts) > max_col_idx:
                        data_started = True
                        try:
                            comp_id_idx = atom_site_cols.get('label_comp_id')
                            chain_idx = atom_site_cols.get('label_asym_id')
                            seq_id_idx = atom_site_cols.get('label_seq_id')
                            
                            if comp_id_idx is None or chain_idx is None or seq_id_idx is None:
                                continue
                            
                            if comp_id_idx >= len(parts) or chain_idx >= len(parts) or seq_id_idx >= len(parts):
                                continue
                            
                            comp_id = parts[comp_id_idx].strip()
                            chain = parts[chain_idx].strip()
                            seq_id_str = parts[seq_id_idx].strip()
                            
                            if chain == chain_id:
                                try:
                                    seq_id = int(seq_id_str)
                                    residue_key = f"{chain}_{seq_id}"
                                    
                                    if residue_key not in seen_residues:
                                        seen_residues.add(residue_key)
                                        
                                        atom_name_idx = atom_site_cols.get('label_atom_id', -1)
                                        if atom_name_idx >= 0 and atom_name_idx < len(parts):
                                            atom_name = parts[atom_name_idx].strip()
                                            if atom_name == 'CA':
                                                aa_single = three_to_one_letter(comp_id)
                                                if aa_single:
                                                    sequence_dict[seq_id] = aa_single
                                        else:
                                            
                                            aa_single = three_to_one_letter(comp_id)
                                            if aa_single:
                                                sequence_dict[seq_id] = aa_single
                                except (ValueError, IndexError):
                                    continue
                        except (IndexError, ValueError):
                            continue
        
        
        if sequence_dict:
            sorted_seq_ids = sorted(sequence_dict.keys())
            sequence = ''.join([sequence_dict[seq_id] for seq_id in sorted_seq_ids])
            return sequence, None
        else:
            return None, f"chain {chain_id} nottoorsequenceasempty"
    
    except Exception as e:
        return None, f"parseerror: {str(e)}"


def create_monomer_json(sequence_id: str, sequence: str, chain_id: str = 'A', 
                        model_seeds: Optional[List[int]] = None) -> dict:
    """create monomer json."""
    if model_seeds is None:
        model_seeds = [1]
    
    af3_json = {
        "dialect": "alphafold3",
        "version": 3,
        "name": sequence_id,
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence,
                    "modifications": [],
                    "unpairedMsa": "",      
                    "pairedMsa": "",        
                    "templates": []         
                }
            }
        ],
        "modelSeeds": model_seeds
    }
    
    return af3_json


def find_cif_files(input_paths: List[str]) -> List[Path]:
    """find cif files."""
    cif_files = []
    
    for input_path in input_paths:
        path = Path(input_path)
        
        if not path.exists():
            print(f"warning: notexists {input_path}")
            continue
            
        if path.is_dir():
            
            found_files = list(path.glob("*.cif")) + list(path.glob("*.mmcif"))
            if not found_files:
                print(f"warning: directoryinnotoCIFfile {input_path}")
            else:
                cif_files.extend(found_files)
                print(f"indirectory {input_path} into {len(found_files)} CIFfile")
        elif path.is_file() and path.suffix.lower() in ['.cif', '.mmcif']:
            
            cif_files.append(path)
        else:
            print(f"warning: notsupportfile {input_path}")
    
    
    cif_files = sorted(set(cif_files))
    return cif_files


def main():
    parser = argparse.ArgumentParser(
        description='fromCIFfileextractsequencegenerateAF3JSONfile',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n:\n # asCIFfilegenerateJSON\n python create_af3_json_from_cif.py \\\n --input structure.cif \\\n --chain_id A \\\n --output_dir af3_monomer_inputs\n \n # asdirectoryinallCIFfilegenerateJSON\n python create_af3_json_from_cif.py \\\n --input./cif_files/ \\\n --chain_id B \\\n --output_dir af3_monomer_inputs \\\n --model_seeds 1 2 3\n \n # input\n python create_af3_json_from_cif.py \\\n --input file1.cif file2.cif./dir1/./dir2/ \\\n --chain_id A \\\n --output_dir af3_monomer_inputs\n '
    )
    
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='inputCIFfileorcontainsCIFfiledirectory()')
    parser.add_argument('--output_dir', '-o', required=True,
                       help='outputJSONfiledirectory')
    parser.add_argument('--chain_id', '-c', default='A',
                       help='extractchainID(default: A)')
    parser.add_argument('--model_seeds', type=int, nargs='+', default=[1],
                       help='(default: 1)')
    parser.add_argument('--name_prefix', default='',
                       help='JSONfile(optional)')
    parser.add_argument('--skip_empty', action='store_true',
                       help='skipsequenceasemptyfile')
    
    args = parser.parse_args()
    
    print("="*80)
    print('AF3 JSON generate(fromCIFfile)')
    print("="*80)
    print(f"input: {args.input}")
    print(f"output directory: {args.output_dir}")
    print(f"chain ID: {args.chain_id}")
    print(f": {args.model_seeds}")
    print()
    
    
    print('CIFfile...')
    cif_files = find_cif_files(args.input)
    
    if not cif_files:
        print('error: notoCIFfile')
        sys.exit(1)
    
    print(f"to {len(cif_files)} CIFfile\n")
    
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    print('extractsequencegenerateJSONfile...')
    generated_files = []
    failed_files = []
    
    for i, cif_file in enumerate(cif_files, 1):
        print(f"[{i}/{len(cif_files)}]: {cif_file.name}")
        
        
        sequence, error = extract_sequence_from_cif(cif_file, args.chain_id)
        
        if sequence is None:
            error_msg = error or 'noterror'
            print(f" [ERROR] extractsequencefailed: {error_msg}")
            if not args.skip_empty:
                failed_files.append((cif_file, error_msg))
            continue
        
        if len(sequence) == 0:
            print(f" [WARN] sequenceasempty")
            if not args.skip_empty:
                failed_files.append((cif_file, 'sequenceasempty'))
            continue
        
        print(f" [OK] extractsequenceok,length: {len(sequence)}")
        
        
        seq_id = cif_file.stem
        if args.name_prefix:
            seq_id = f"{args.name_prefix}_{seq_id}"
        
        
        af3_json = create_monomer_json(
            sequence_id=seq_id,
            sequence=sequence,
            chain_id=args.chain_id,
            model_seeds=args.model_seeds
        )
        
        
        safe_name = seq_id.replace('/', '_').replace(',', '_').replace(' ', '_')
        safe_name = safe_name[:50]  
        json_file = output_dir / f"{safe_name}_monomer.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(af3_json, f, indent=2)
        
        generated_files.append((safe_name, seq_id, json_file, len(sequence)))
        print(f" [OK] JSONfilealreadygenerate: {json_file.name}\n")
    
    
    mapping_file = output_dir / "cif_id_mapping.txt"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write('AF3 - CIFfile\n')
        f.write("="*80 + "\n\n")
        f.write(f"chainID: {args.chain_id}\n")
        f.write(f"ok: {len(generated_files)} file\n")
        if failed_files:
            f.write(f"failed: {len(failed_files)} file\n")
        f.write("\n")
        f.write(f"{'JSON file':<50} | {'CIFfile':<40} | sequencelength\n")
        f.write("-"*100 + "\n")
        for safe_name, orig_id, json_file, seq_len in generated_files:
            f.write(f"{safe_name}_monomer.json {' '*(48-len(safe_name))} | {orig_id:<40} | {seq_len}\n")
        
        if failed_files:
            f.write("\n" + "="*80 + "\n")
            f.write('failedfile:\n')
            f.write("="*80 + "\n")
            for cif_file, error in failed_files:
                f.write(f"{cif_file.name:<50} | {error}\n")
    
    print("="*80)
    print('[SUCCESS] ！')
    print("="*80)
    print(f"\ngeneratefile:")
    print(f"  {len(generated_files)} JSON filein: {args.output_dir}/")
    print(f": {mapping_file}")
    if failed_files:
        print(f" failed: {len(failed_files)} file(info)")
    print("="*80)


if __name__ == "__main__":
    main()

