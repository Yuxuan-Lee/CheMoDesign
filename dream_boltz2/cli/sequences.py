#!/usr/bin/env python3
"""Design sequences with LigandMPNN on screened or woken backbones."""

import sys
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict


try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dream_boltz2.sequence.ligandmpnn import LigandMPNNRunner
from dream_boltz2.io.pdb import parse_pdb_chain


def parse_fixed_residues_from_yaml(
    yaml_path: str,
    binder_chains: Optional[List[str]] = None,
    receptor_chains: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """parse fixed residues from yaml."""
    binder_fixed = defaultdict(list)
    receptor_fixed = defaultdict(list)
    
    if not HAS_YAML:
        raise ImportError('needpyyamlfromYAMLreadfixedresidue.: pip install pyyaml')
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        
        if not binder_chains and 'dream' in config:
            dream_config = config['dream']
            if 'binder_chains' in dream_config:
                binder_chains = dream_config['binder_chains']
        if not receptor_chains and 'dream' in config:
            dream_config = config['dream']
            if 'receptor_chains' in dream_config:
                receptor_chains = dream_config['receptor_chains']
        
        if 'constraints' not in config:
            return dict(binder_fixed), dict(receptor_fixed)
        
        constraints = config.get('constraints', [])
        
        for constraint in constraints:
            if 'bond' not in constraint:
                continue
            
            bond = constraint['bond']
            atom1 = bond.get('atom1', [])
            atom2 = bond.get('atom2', [])
            
            
            if len(atom1) >= 2:
                chain1 = atom1[0]
                res_num1 = atom1[1]
                
                
                
                
                if binder_chains and chain1 in binder_chains:
                    if res_num1 not in binder_fixed[chain1]:
                        binder_fixed[chain1].append(res_num1)
                elif receptor_chains and chain1 in receptor_chains:
                    if res_num1 not in receptor_fixed[chain1]:
                        receptor_fixed[chain1].append(res_num1)
                
                
            
            
            if len(atom2) >= 2:
                chain2 = atom2[0]
                res_num2 = atom2[1]
                
                
                
                if binder_chains and chain2 in binder_chains:
                    if res_num2 not in binder_fixed[chain2]:
                        binder_fixed[chain2].append(res_num2)
                elif receptor_chains and chain2 in receptor_chains:
                    if res_num2 not in receptor_fixed[chain2]:
                        receptor_fixed[chain2].append(res_num2)
                
        
        
        for chain in binder_fixed:
            binder_fixed[chain] = sorted(binder_fixed[chain])
        for chain in receptor_fixed:
            receptor_fixed[chain] = sorted(receptor_fixed[chain])
        
    except Exception as e:
        print(f" readYAMLconfigfilefailed {yaml_path}: {e}")
        return dict(binder_fixed), dict(receptor_fixed)
    
    return dict(binder_fixed), dict(receptor_fixed)


def detect_chains_in_pdb(pdb_path: str) -> Dict[str, int]:
    """detect chains in pdb."""
    chain_info = {}
    
    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        
        current_chain = None
        current_res_num = None
        chain_residues = {}
        
        for line in lines:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            
            chain = line[21:22].strip()
            if not chain:
                continue
            
            try:
                res_num = int(line[22:26].strip())
            except (ValueError, IndexError):
                continue
            
            
            if chain not in chain_residues:
                chain_residues[chain] = set()
            chain_residues[chain].add(res_num)
        
        
        for chain, residues in chain_residues.items():
            chain_info[chain] = len(residues)
    
    except Exception as e:
        print(f" readPDBfilefailed {pdb_path}: {e}")
        return {}
    
    return chain_info


def detect_fixed_residues_from_bfactor(
    pdb_path: str,
    binder_chain: str,
    bfactor_threshold: float = 0.99
) -> List[int]:
    """detect fixed residues from bfactor."""
    fixed_residues = []
    
    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        
        current_res_num = None
        current_chain = None
        current_bfactor = None
        
        for line in lines:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            
            chain = line[21:22].strip()
            res_num = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            
            
            if chain != binder_chain:
                continue
            
            
            if atom_name == 'CA':
                try:
                    bfactor = float(line[60:66].strip())
                except (ValueError, IndexError):
                    continue
                
                
                if bfactor >= bfactor_threshold:
                    if res_num not in fixed_residues:
                        fixed_residues.append(res_num)
    
    except Exception as e:
        print(f" readPDBfilefailed {pdb_path}: {e}")
        return []
    
    return sorted(fixed_residues)


def format_fixed_residues(fixed_residues: List[int], chain: str) -> str:
    """format fixed residues."""
    return " ".join([f"{chain}{res_num}" for res_num in fixed_residues])


def generate_fixed_positions_jsonl(
    pdb_dir: Path,
    binder_chain: str,
    receptor_chain: Optional[str] = None,
    receptor_fixed_residues: Optional[List[int]] = None,
    binder_fixed_residues: Optional[List[int]] = None,
    bfactor_threshold: float = 0.99,
    yaml_config: Optional[str] = None,
    binder_chains: Optional[List[str]] = None,
    receptor_chains: Optional[List[str]] = None,
    output_file: Optional[Path] = None
) -> Dict[str, Dict[str, List[int]]]:
    """generate fixed positions jsonl."""
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    fixed_positions_dict = {}
    
    
    yaml_binder_fixed = {}
    yaml_receptor_fixed = {}
    if yaml_config and Path(yaml_config).exists():
        print(f"\n{'='*70}")
        print(' fromYAMLconfigfilereadfixedresidue(fromconstraints)')
        print("="*70)
        yaml_binder_fixed, yaml_receptor_fixed = parse_fixed_residues_from_yaml(
            yaml_config,
            binder_chains=binder_chains,
            receptor_chains=receptor_chains
        )
        
        if yaml_binder_fixed:
            print(f"fromYAMLreadBinderfixedresidue:")
            for chain, residues in yaml_binder_fixed.items():
                print(f"  {chain}chain: {residues}")
        
        if yaml_receptor_fixed:
            print(f"fromYAMLreadreceptorfixedresidue:")
            for chain, residues in yaml_receptor_fixed.items():
                print(f"  {chain}chain: {residues}")
        
        if not yaml_binder_fixed and not yaml_receptor_fixed:
            print(f" YAMLfileinnottoconstraintinfo")
        print("="*70)
    
    print(f"\n{'='*70}")
    print(' fixedresidue')
    print(f"{'='*70}")
    print(f"PDBdirectory: {pdb_dir}")
    print(f"to {len(pdb_files)} PDBfile")
    if yaml_config:
        print(f"YAMLconfig: {yaml_config}")
    print(f"B-factorthreshold: {bfactor_threshold}")
    print(f"Binderchain: {binder_chain} (PDBfileinchainID)")
    if binder_fixed_residues:
        print(f"defaultBinderfixedresidue(allPDB): {binder_fixed_residues}")
    if receptor_chain and receptor_fixed_residues:
        print(f"receptorchain: {receptor_chain}, fixedresidue: {receptor_fixed_residues}")
    
    
    if yaml_binder_fixed and binder_chain:
        yaml_binder_chains = list(yaml_binder_fixed.keys())
        if binder_chain not in yaml_binder_chains:
            print(f"\n warning: --binder_chain '{binder_chain}' notinYAMLbinderfixedresiduechainin！")
            print(f" YAMLinbinderchain: {yaml_binder_chains}")
            print(f" thisnofromYAMLreadfixedresidue")
            print(f":")
            print(f" 1. ifPDBinbinderchain '{binder_chain}',use --binder_fixed_residues ")
            print(f" 2. ifYAMLinbinderchain {yaml_binder_chains[0]},checkPDBfileinchainID")
            print(f" 3. autofromYAMLread,needchainIDmatch")
    
    print("="*70)
    print()
    
    for pdb_file in pdb_files:
        pdb_name = pdb_file.stem
        
        
        binder_fixed = []
        if yaml_binder_fixed:
            
            binder_fixed = yaml_binder_fixed.get(binder_chain, []).copy()
            
            
            if not binder_fixed and yaml_binder_fixed:
                available_chains = list(yaml_binder_fixed.keys())
                if len(available_chains) == 1:
                    
                    yaml_chain = available_chains[0]
                    binder_fixed = yaml_binder_fixed[yaml_chain].copy()
                    if len(fixed_positions_dict) < 3:  
                        print(f" auto: YAMLinbinderchain '{yaml_chain}',fixedresidue {binder_fixed} toPDBchain '{binder_chain}'")
                elif len(fixed_positions_dict) < 3:  
                    print(f" binderchain '{binder_chain}' inYAMLinnofixedresidue")
                    print(f" YAMLinbinderchain: {available_chains}")
                    print(f": use --binder_chain {available_chains[0] if available_chains else '?'}")
        
        
        
        bfactor_fixed = detect_fixed_residues_from_bfactor(
            str(pdb_file),
            binder_chain,
            bfactor_threshold
        )
        
        
        if bfactor_fixed:
            binder_fixed = sorted(list(set(binder_fixed + bfactor_fixed)))
            if len(fixed_positions_dict) < 3 and bfactor_fixed:  
                print(f" B-factor: to {len(bfactor_fixed)} fixedresidue {bfactor_fixed[:5]}{'...' if len(bfactor_fixed) > 5 else ''}")
        
        
        if binder_fixed_residues:
            binder_fixed = sorted(list(set(binder_fixed + binder_fixed_residues)))
        
        
        fixed_dict = {binder_chain: sorted(binder_fixed)}
        
        
        
        if receptor_chain:
            receptor_fixed = []
            if yaml_receptor_fixed:
                
                receptor_fixed = yaml_receptor_fixed.get(receptor_chain, []).copy()
                
                
                if not receptor_fixed and yaml_receptor_fixed:
                    available_chains = list(yaml_receptor_fixed.keys())
                    if len(available_chains) == 1:
                        
                        yaml_chain = available_chains[0]
                        receptor_fixed = yaml_receptor_fixed[yaml_chain].copy()
                        if len(fixed_positions_dict) < 3:  
                            print(f" auto: YAMLinreceptorchain '{yaml_chain}',fixedresidue {receptor_fixed} toPDBchain '{receptor_chain}'")
            
            
            if not receptor_fixed and receptor_fixed_residues:
                receptor_fixed = receptor_fixed_residues.copy()
            
            
            if receptor_fixed:
                fixed_dict[receptor_chain] = sorted(receptor_fixed)
        
        fixed_positions_dict[pdb_name] = fixed_dict
        
        
        try:
            binder_seq, _, _ = parse_pdb_chain(str(pdb_file), binder_chain, verbose=False)
            binder_length = len(binder_seq)
            designable_count = binder_length - len(binder_fixed)
        except:
            binder_length = None
            designable_count = None
        
        if binder_fixed:
            print(f"  {pdb_name}:")
            print(f" - Binderlength: {binder_length if binder_length else 'not'}")
            print(f" - fixedresidue: {len(binder_fixed)} {binder_fixed[:5]}{'...' if len(binder_fixed) > 5 else ''}")
            print(f" - residue: {designable_count if designable_count else 'not'} ")
        else:
            print(f"  {pdb_name}: nottofixedresidue(B-factor < {bfactor_threshold}),")
    
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for pdb_name, fixed_dict in fixed_positions_dict.items():
                entry = {pdb_name: fixed_dict}
                f.write(json.dumps(entry) + '\n')
        print(f"\n fixed_positions.jsonl alreadysaveto: {output_file}")
    
    return fixed_positions_dict


def batch_design_sequences(
    input_dir: Path,
    binder_chain: str,
    num_sequences: int = 200,
    temperature: float = 0.1,
    output_dir: Optional[Path] = None,
    bias_AA_json: Optional[str] = None,
    bias_AA: Optional[str] = None,
    model_type: str = "auto",
    receptor_chain: Optional[str] = None,
    receptor_fixed_residues: Optional[List[int]] = None,
    binder_fixed_residues: Optional[List[int]] = None,
    bfactor_threshold: float = 0.99,
    yaml_config: Optional[str] = None,
    binder_chains: Optional[List[str]] = None,
    receptor_chains: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """batch design sequences."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"inputdirectorynotexists: {input_dir}")
    
    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        print(f" nottoPDBfile: {input_dir}")
        return
    
    
    if output_dir is None:
        output_dir = input_dir.parent / "ligandmpnn_sequences"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    fixed_positions_file = output_dir / "fixed_positions.jsonl"
    fixed_positions_dict = generate_fixed_positions_jsonl(
        input_dir,
        binder_chain,
        receptor_chain=receptor_chain,
        receptor_fixed_residues=receptor_fixed_residues,
        binder_fixed_residues=binder_fixed_residues,
        bfactor_threshold=bfactor_threshold,
        yaml_config=yaml_config,
        binder_chains=binder_chains,
        receptor_chains=receptor_chains,
        output_file=fixed_positions_file
    )
    
    
    print(f"\n{'='*70}")
    print(' initLigandMPNN Runner')
    print("="*70)
    try:
        runner = LigandMPNNRunner(
            model_type=model_type,
            device='cuda'
        )
        print(f" LigandMPNN Runner initok")
        print(f": {model_type}")
    except Exception as e:
        print(f" initfailed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    
    first_pdb_chains = {}
    if pdb_files:
        first_pdb_chains = detect_chains_in_pdb(str(pdb_files[0]))
    
    
    print(f"\n{'='*70}")
    print(f" sequence")
    print("="*70)
    print(f"PDBfile: {len(pdb_files)}")
    print(f"eachPDBgenerate: {num_sequences} sequence")
    print(f"sequence: {len(pdb_files) * num_sequences}")
    print(f"sampling: {temperature} {'()' if temperature < 0.15 else '(in)' if temperature < 0.5 else '()'}")
    if seed is not None:
        print(f": {seed} ()")
    else:
        print(f": autogenerate (timesnot)")
    if bias_AA_json:
        print(f": {bias_AA_json} (JSONfile)")
    elif bias_AA:
        print(f": {bias_AA} ()")
    else:
        print(f": no")
    
    
    print(f"\n:")
    print(f" chain: {binder_chain} (chain)")
    if receptor_chain:
        print(f" fixedchain: {receptor_chain} (chainfixed,not)")
    print(f" chain: autofixed(not)")
    
    
    print(f"\n chainID:")
    if yaml_config and Path(yaml_config).exists():
        print(f" YAMLconfig:")
        if binder_chains:
            print(f" Binderchain: {binder_chains}")
        if receptor_chains:
            print(f" Receptorchain: {receptor_chains}")
    print(f" argument:")
    print(f"    --binder_chain: {binder_chain}")
    if receptor_chain:
        print(f"    --receptor_chain: {receptor_chain}")
    if first_pdb_chains:
        print(f" PDBfileinchainID: {first_pdb_chains}")
        
        if binder_chain not in first_pdb_chains:
            print(f" warning: binderchain '{binder_chain}' notinPDBfilein！")
        if receptor_chain and receptor_chain not in first_pdb_chains:
            print(f" warning: receptorchain '{receptor_chain}' notinPDBfilein！")
    
    print("="*70)
    print()
    
    all_sequences = []
    success_count = 0
    
    for i, pdb_file in enumerate(pdb_files, 1):
        pdb_name = pdb_file.stem
        print(f"[{i}/{len(pdb_files)}]: {pdb_name}")
        
        
        if pdb_name in fixed_positions_dict:
            fixed_dict = fixed_positions_dict[pdb_name]
            binder_fixed = fixed_dict.get(binder_chain, [])
            
            
            if not binder_fixed and fixed_dict:
                available_chains = list(fixed_dict.keys())
                print(f" warning: binderchain '{binder_chain}' infixed_positions_dictinnottofixedresidue")
                print(f" chain: {available_chains}")
                print(f" thischainfixedresidue: {[(k, v) for k, v in fixed_dict.items()]}")
            
            if binder_fixed:
                fixed_residues_str = format_fixed_residues(binder_fixed, binder_chain)
            else:
                fixed_residues_str = None
            
            
            if receptor_chain:
                if receptor_chain in fixed_dict:
                    receptor_fixed = fixed_dict[receptor_chain]
                    if receptor_fixed:
                        receptor_fixed_str = format_fixed_residues(receptor_fixed, receptor_chain)
                        if fixed_residues_str:
                            fixed_residues_str += " " + receptor_fixed_str
                        else:
                            fixed_residues_str = receptor_fixed_str
                else:
                    
                    try:
                        receptor_seq, _, _ = parse_pdb_chain(str(pdb_file), receptor_chain, verbose=False)
                        receptor_all_residues = list(range(1, len(receptor_seq) + 1))
                        receptor_fixed_str = format_fixed_residues(receptor_all_residues, receptor_chain)
                        if fixed_residues_str:
                            fixed_residues_str += " " + receptor_fixed_str
                        else:
                            fixed_residues_str = receptor_fixed_str
                        if i <= 3:  
                            print(f" fixedreceptorchain {receptor_chain}({len(receptor_all_residues)} residue,not)")
                    except Exception as e:
                        if i <= 3:
                            print(f" noreadreceptorchain {receptor_chain},noautofixed: {e}")
            
            if fixed_residues_str:
                
                binder_info = f"Binderchain{binder_chain}: {len(binder_fixed)} fixedresidue"
                if binder_fixed:
                    binder_info += f" {binder_fixed}"
                
                receptor_info = ""
                if receptor_chain and receptor_chain in fixed_dict:
                    receptor_fixed = fixed_dict[receptor_chain]
                    if receptor_fixed:
                        receptor_info = f", Receptorchain{receptor_chain}: {len(receptor_fixed)} fixedresidue {receptor_fixed}"
                
                print(f" fixedresidue: {binder_info}{receptor_info}")
                
                
                try:
                    binder_seq, _, _ = parse_pdb_chain(str(pdb_file), binder_chain, verbose=False)
                    total_residues = len(binder_seq)
                    designable_count = total_residues - len(binder_fixed)
                    print(f" residue: {designable_count}/{total_residues} ")
                except:
                    pass
            else:
                print(f" nottofixedresidueinfo()")
                fixed_residues_str = None
        else:
            fixed_residues_str = None
            print(f" nottofixedresidueinfo(pdb_namenotinfixed_positions_dictin)")
        
        
        pdb_output_dir = output_dir / pdb_name
        
        try:
            
            
            
            
            MAX_BATCH_SIZE = 1000
            effective_batch_size = batch_size if batch_size is not None else num_sequences
            if effective_batch_size > MAX_BATCH_SIZE:
                print(
                    f" batch_size={effective_batch_size} {MAX_BATCH_SIZE},alreadyautoas {MAX_BATCH_SIZE}"
                )
                effective_batch_size = MAX_BATCH_SIZE
            
            gen_kwargs = {
                'pdb_path': str(pdb_file),
                'binder_chain': binder_chain,
                'num_sequences': num_sequences,
                'temperature': temperature,
                'fixed_residues': fixed_residues_str,
                'out_dir': str(pdb_output_dir),
                'batch_size': effective_batch_size,
                'seed': seed,
            }
            
            
            import inspect
            sig = inspect.signature(runner.generate_sequences)
            if 'bias_AA' in sig.parameters:
                gen_kwargs['bias_AA'] = bias_AA
            if 'bias_AA_json' in sig.parameters:
                gen_kwargs['bias_AA_json'] = bias_AA_json
            
            sequences = runner.generate_sequences(**gen_kwargs)
            
            
            fasta_file = pdb_output_dir / "sequences.fasta"
            with open(fasta_file, 'w', encoding='utf-8') as f:
                for j, seq in enumerate(sequences, 1):
                    f.write(f">{pdb_name}_seq_{j:06d}\n{seq}\n")
            
            all_sequences.extend([(pdb_name, seq) for seq in sequences])
            success_count += 1
            
            unique_count = len(set(sequences))
            print(f" ok: {len(sequences)} sequence({unique_count} )")
            
        except Exception as e:
            print(f" failed: {e}")
            import traceback
            traceback.print_exc()
    
    
    merged_fasta = output_dir / "all_sequences.fasta"
    with open(merged_fasta, 'w', encoding='utf-8') as f:
        for pdb_name, seq in all_sequences:
            f.write(f">{pdb_name}\n{seq}\n")
    
    
    print(f"\n{'='*70}")
    print(' ')
    print("="*70)
    print(f"ok: {success_count}/{len(pdb_files)} PDBfile")
    print(f"sequence: {len(all_sequences)}")
    print(f"sequence: {len(set(seq for _, seq in all_sequences))}")
    print(f"\noutputfile:")
    print(f"  - fixed_positions.jsonl: {fixed_positions_file}")
    print(f"  - all_sequences.fasta: {merged_fasta}")
    print(f" - eachPDBsequence: {output_dir}/*/sequences.fasta")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Design binder sequences with LigandMPNN.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python -m dream_boltz2.cli.sequences --input_dir screen_results --binder_chain B --num_sequences 50 --model_type soluble_mpnn"

    )
    
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='inputPDBfiledirectory')
    parser.add_argument('--binder_chain', type=str, required=True,
                       help='BinderchainID( B)')
    
    
    parser.add_argument('--num_sequences', type=int, default=200,
                       help='eachPDBgeneratesequence(default: 200)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='sampling(default: 0.1,: 0.05-0.8,,)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='output directory(default: inputdirectorydirectory/ligandmpnn_sequences)')
    
    
    parser.add_argument('--bfactor_threshold', type=float, default=0.99,
                       help='B-factorthreshold(default: 0.99,B-factor=1.0fixed)')
    parser.add_argument('--receptor_chain', type=str, default=None,
                       help='receptorchainID(optional)')
    parser.add_argument('--receptor_fixed_residues', type=str, default=None,
                       help='receptorfixedresidue(, "41,42",allPDB)')
    parser.add_argument('--binder_fixed_residues', type=str, default=None,
                       help='defaultBinderfixedresidue(, "30,31",allPDB,andYAML/B-factor)')
    
    
    parser.add_argument('--yaml_config', type=str, default=None,
                       help='YAMLconfigfile path(fromconstraintsinautoextractfixedresidue,B-factor)')
    parser.add_argument('--binder_chains', type=str, default=None,
                       help='BinderchainIDlist(,forYAMLparse, "B")')
    parser.add_argument('--receptor_chains', type=str, default=None,
                       help='receptorchainIDlist(,forYAMLparse, "A")')
    
    
    parser.add_argument('--bias_AA_json', type=str, default=None,
                       help='JSONfile path(,: {"global": {"D": 1.5, "E": 1.5,...}})')
    parser.add_argument('--bias_AA', type=str, default=None,
                       help='(: "D:1.5,E:1.5,K:1.0,C:-5.0")')
    
    
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['auto', 'ligand_mpnn', 'soluble_mpnn', 'protein_mpnn'],
                       help='(default: auto)')
    
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='LigandMPNN roundgeneratesequence(default: 16).; OOM , 8 or 4')
    parser.add_argument('--seed', type=int, default=None,
                       help='(default: None,use;for, --seed 42)')
    
    args = parser.parse_args()
    
    
    receptor_fixed_residues = None
    if args.receptor_fixed_residues:
        try:
            receptor_fixed_residues = [int(x.strip()) for x in args.receptor_fixed_residues.split(',')]
        except ValueError:
            print(f"[ERROR] noreceptorfixedresidue: {args.receptor_fixed_residues}")
            return 1
    
    binder_fixed_residues = None
    if args.binder_fixed_residues:
        try:
            binder_fixed_residues = [int(x.strip()) for x in args.binder_fixed_residues.split(',')]
        except ValueError:
            print(f"[ERROR] noBinderfixedresidue: {args.binder_fixed_residues}")
            return 1
    
    
    binder_chains = None
    if args.binder_chains:
        binder_chains = [c.strip() for c in args.binder_chains.split(',')]
    
    receptor_chains = None
    if args.receptor_chains:
        receptor_chains = [c.strip() for c in args.receptor_chains.split(',')]
    
    
    if args.yaml_config and Path(args.yaml_config).exists():
        if not HAS_YAML:
            print(f" needpyyamlfromYAMLreadchainID.: pip install pyyaml")
        else:
            try:
                with open(args.yaml_config, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                
                if 'dream' in config:
                    dream_config = config['dream']
                    if not binder_chains and 'binder_chains' in dream_config:
                        binder_chains = dream_config['binder_chains']
                    if not receptor_chains and 'receptor_chains' in dream_config:
                        receptor_chains = dream_config['receptor_chains']
            except Exception as e:
                print(f" fromYAMLreadchainIDfailed: {e}")
    
    
    try:
        batch_design_sequences(
            input_dir=Path(args.input_dir),
            binder_chain=args.binder_chain,
            num_sequences=args.num_sequences,
            temperature=args.temperature,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            bias_AA_json=args.bias_AA_json,
            bias_AA=args.bias_AA,
            model_type=args.model_type,
            receptor_chain=args.receptor_chain,
            receptor_fixed_residues=receptor_fixed_residues,
            binder_fixed_residues=binder_fixed_residues,
            bfactor_threshold=args.bfactor_threshold,
            yaml_config=args.yaml_config,
            binder_chains=binder_chains,
            receptor_chains=receptor_chains,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

