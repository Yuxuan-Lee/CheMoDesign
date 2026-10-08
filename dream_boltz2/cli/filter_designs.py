#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep designs that pass confidence thresholds."""

import os
import argparse
import random
import re
from pathlib import Path

def parse_fasta_file(filepath):
    """parse fasta file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        header, seq = '', ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and seq:
                    yield header, seq
                header = line
                seq = ''
            else:
                seq += line
        if header and seq:
            yield header, seq

def get_score_from_header(header, score_type='overall_confidence', sequence=None):
    """get score from header."""
    
    if score_type == 'min_ala':
        if sequence:
            seq_upper = sequence.upper()
            
            for sep in ['/', ':']:
                if sep in seq_upper:
                    parts = seq_upper.split(sep)
                    
                    seq_upper = min(parts, key=len) if len(parts) > 1 else seq_upper
                    break
            ala_count = seq_upper.count('A')
            return ala_count / max(len(seq_upper), 1)  
        return float('inf')

    
    if score_type in ['overall_confidence', 'ligand_confidence']:
        pattern = rf'{score_type}=([0-9\.]+)'
        match = re.search(pattern, header)
        if match:
            
            return -float(match.group(1))

    elif score_type == 'seq_rec':
        match = re.search(r'seq_rec=([0-9\.]+)', header)
        if match:
            return -float(match.group(1))

    
    elif score_type == 'score':
        match = re.search(r'score=([0-9\.]+)', header)
        if match:
            return float(match.group(1))

    
    return float('inf')

def clean_sequence(sequence, keep_part='binder', receptor_first=False):
    """clean sequence."""
    sep = None
    if ':' in sequence:
        sep = ':'
    elif '/' in sequence:
        sep = '/'

    if sep:
        parts = sequence.split(sep)
        
        if receptor_first:
            
            if keep_part == 'binder' and len(parts) >= 2:
                return parts[1].strip().upper()
            elif keep_part == 'receptor' and len(parts) >= 1:
                return parts[0].strip().upper()
            else:  # 'both'
                return sequence.replace(sep, '').upper()
        else:
            
            if sep == ':':
                
                if keep_part == 'binder' and len(parts) >= 1:
                    return parts[0].strip().upper()
                elif keep_part == 'receptor' and len(parts) >= 2:
                    return parts[1].strip().upper()
                else:
                    return sequence.replace(sep, '').upper()
            else:
                
                if keep_part == 'binder' and len(parts) >= 2:
                    return parts[1].strip().upper()
                elif keep_part == 'receptor' and len(parts) >= 1:
                    return parts[0].strip().upper()
                else:
                    return sequence.replace(sep, '').upper()

    return sequence.upper()

def detect_receptor_binder_order(sequences, sep=None):
    """detect receptor binder order."""
    parts0, parts1 = [], []
    for seq in sequences:
        s = seq.strip()
        if sep is None:
            if ':' in s:
                sep_used = ':'
            elif '/' in s:
                sep_used = '/'
            else:
                continue
        else:
            sep_used = sep
        if sep_used not in s:
            continue
        p = s.split(sep_used, 1)
        if len(p) != 2:
            continue
        parts0.append(p[0].strip())
        parts1.append(p[1].strip())
    if not parts0 or not parts1 or len(parts0) != len(parts1):
        return None
    n_unique_0 = len(set(parts0))
    n_unique_1 = len(set(parts1))
    n = len(parts0)
    
    if n_unique_0 <= 1 and n_unique_1 > 1:
        return True   
    if n_unique_1 <= 1 and n_unique_0 > 1:
        return False  
    if n_unique_0 < n_unique_1:
        return True
    if n_unique_1 < n_unique_0:
        return False
    return None  


def find_seqs_directories(outputs_dir):
    """find seqs directories."""
    outputs_path = Path(outputs_dir)
    seqs_dirs = []
    
    
    for task_dir in outputs_path.iterdir():
        if task_dir.is_dir():
            seqs_dir = task_dir / 'seqs'
            if seqs_dir.exists() and seqs_dir.is_dir():
                seqs_dirs.append((task_dir.name, seqs_dir))
    
    return seqs_dirs

def main():
    """main."""
    parser = argparse.ArgumentParser(
        description='screenLigandMPNNsequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n:\n # seqsdirectory\n python filter_best_designs_v2.py --source_folder outputs/task_1/seqs --output_folder results/ --top_n 20\n \n # outputsall()\n python filter_best_designs_v2.py --outputs_dir outputs/ --output_folder results/ --top_n 20\n \n # useligand\n python filter_best_designs_v2.py --outputs_dir outputs/ --output_folder results/ --score_type ligand_confidence\n '
    )
    
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--source_folder', type=str, 
                            help='seqsfile')
    input_group.add_argument('--outputs_dir', type=str,
                            help='outputsdirectory(autoalldirectoryseqs)')
    
    
    parser.add_argument('--output_folder', type=str, required=True,
                       help='outputfile')
    parser.add_argument('--top_n', type=int, default=20,
                       help='eachtopsequence(default:20)')
    parser.add_argument('--score_type', type=str,
                       default='overall_confidence',
                       choices=['overall_confidence', 'ligand_confidence', 'seq_rec', 'score', 'min_ala', 'random'],
                       help='(default:overall_confidence).min_ala: A;random: FASTA')
    parser.add_argument('--random-seed', type=int, default=None, dest='random_seed',
                       help=' --score_type random:,')
    parser.add_argument('--remove_slash', action='store_true',
                       help='sequencein(defaultbinder)')
    parser.add_argument('--keep_part', type=str, default='binder',
                       choices=['binder', 'both', 'receptor'],
                       help='/sequence:binder(binder), both(), receptor(receptor)')
    parser.add_argument('--receptor-first', action='store_true',
                       help='sequenceas「receptorin, binderin」(receptor:binder or receptor/binder).'
                            'LigandMPNN output round01_sample10_*_processed.fa asthis,argument binder.')
    parser.add_argument('--auto-detect-order', action='store_true',
                       help='autoreceptor/binder:vsfilesequence,asreceptor, as binder.'
                            'and --remove_slash --receptor-first.')
    parser.add_argument('--output_filename', type=str, default='best_designs.fa',
                       help='outputfile(default:best_designs.fa)')
    parser.add_argument('--rename', action='store_true',
                       help='sequence(generateID,task1_bb2_seq3)')
    parser.add_argument('--name_map', action='store_true',
                       help='savefile(need--rename)')
    
    args = parser.parse_args()

    if args.score_type == 'random' and args.random_seed is not None:
        random.seed(args.random_seed)

    
    os.makedirs(args.output_folder, exist_ok=True)
    output_file_path = os.path.join(args.output_folder, args.output_filename)
    
    print("="*80)
    print('LigandMPNNsequencescreen v2.0')
    print("="*80)
    
    
    seqs_to_process = []
    
    if args.source_folder:
        
        if not os.path.isdir(args.source_folder):
            print(f"error:directorynotexists: {args.source_folder}")
            return
        task_name = Path(args.source_folder).parent.name
        seqs_to_process = [(task_name, Path(args.source_folder))]
        print(f"\nmode:seqsdirectory")
        print(f":{task_name}")
    
    elif args.outputs_dir:
        
        if not os.path.isdir(args.outputs_dir):
            print(f"error:directorynotexists: {args.outputs_dir}")
            return
        seqs_to_process = find_seqs_directories(args.outputs_dir)
        print(f"\nmode:outputsdirectory")
        print(f"to {len(seqs_to_process)} ")
        for task_name, _ in seqs_to_process[:5]:
            print(f"  - {task_name}")
        if len(seqs_to_process) > 5:
            print(f"... {len(seqs_to_process)-5} ")
    
    if not seqs_to_process:
        print('\nerror:nottoseqsdirectory')
        return
    
    print(f"\nconfig:")
    print(f": {args.score_type}")
    if args.score_type == 'random':
        print(f": each FASTA sequence {args.top_n} "
              + (f"(random_seed={args.random_seed})" if args.random_seed is not None else '(no)'))
    print(f" each: Top {args.top_n} sequence")
    print(f": {'' if args.remove_slash else ''}")
    if args.remove_slash:
        print(f": {args.keep_part} ({'binder' if args.keep_part=='binder' else '' if args.keep_part=='both' else 'receptor'})")
    if getattr(args, 'receptor_first', False):
        print(f" sequence: receptorin, binderin (--receptor-first)")
    if getattr(args, 'auto_detect_order', False):
        print(f": auto(=receptor,=binder)")
    print(f" sequence: {'' if args.rename else '()'}")
    if args.rename and args.name_map:
        print(f": (savetoname_mapping.tsv)")
    print(f" outputfile: {output_file_path}")
    
    print("\n" + "="*80)
    print('...')
    print("="*80)
    
    total_tasks = 0
    total_backbones = 0
    total_sequences = 0

    
    cached_detected_order = None
    
    name_mapping = []

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for task_idx, (task_name, seqs_dir) in enumerate(seqs_to_process, 1):
            task_seqs = 0

            
            fa_files = list(seqs_dir.glob('*.fa')) + list(seqs_dir.glob('*.fasta'))

            if not fa_files:
                continue

            total_tasks += 1
            bb_idx = 0  

            
            for fa_file in sorted(fa_files):
                sequences_with_scores = []
                for header, sequence in parse_fasta_file(fa_file):
                    if args.score_type == 'random':
                        sequences_with_scores.append((0.0, header, sequence))
                    else:
                        score = get_score_from_header(header, args.score_type, sequence=sequence)
                        sequences_with_scores.append((score, header, sequence))

                if not sequences_with_scores:
                    continue

                total_backbones += 1
                bb_idx += 1

                
                use_receptor_first = getattr(args, 'receptor_first', False)
                if getattr(args, 'auto_detect_order', False) and args.remove_slash:
                    if cached_detected_order is not None:
                        use_receptor_first = cached_detected_order
                    else:
                        all_seqs = [seq for _, _, seq in sequences_with_scores]
                        detected = detect_receptor_binder_order(all_seqs)
                        if detected is not None:
                            cached_detected_order = detected
                            use_receptor_first = detected
                            print(f" [AUTO] already: {'receptorin (receptor:binder)' if use_receptor_first else 'binderin (binder:receptor)'},file")
                        

                if args.score_type == 'random':
                    random.shuffle(sequences_with_scores)
                else:
                    
                    sequences_with_scores.sort(key=lambda x: x[0])
                top_sequences = sequences_with_scores[:args.top_n]

                
                for seq_idx, (score, header, sequence) in enumerate(top_sequences, 1):
                    if args.remove_slash:
                        sequence = clean_sequence(
                            sequence,
                            keep_part=args.keep_part,
                            receptor_first=use_receptor_first
                        )
                    
                    
                    if args.rename:
                        
                        new_header = f">task{task_idx}_bb{bb_idx}_seq{seq_idx}"
                        
                        
                        if args.name_map:
                            original_header = header.lstrip('>')
                            name_mapping.append((f"task{task_idx}_bb{bb_idx}_seq{seq_idx}", 
                                               original_header, 
                                               task_name, 
                                               fa_file.name))
                    else:
                        
                        original_header = header.lstrip('>')
                        
                        
                        if task_name in original_header:
                            new_header = f">{original_header}"
                        else:
                            new_header = f">{task_name}__{original_header}"
                    
                    outfile.write(f"{new_header}\n{sequence}\n")
                    task_seqs += 1
                    total_sequences += 1
            
            if task_seqs > 0:
                print(f"  [OK] {task_name}: {task_seqs} sequence")
    
    
    if args.rename and args.name_map and name_mapping:
        mapping_file = os.path.join(args.output_folder, 'name_mapping.tsv')
        with open(mapping_file, 'w', encoding='utf-8') as mf:
            mf.write("NewID\tOriginalHeader\tTaskName\tSourceFile\n")
            for new_id, orig_header, task_name, source_file in name_mapping:
                mf.write(f"{new_id}\t{orig_header}\t{task_name}\t{source_file}\n")
        print(f"\n[SAVE] file: {mapping_file}")
    
    print("\n" + "="*80)
    print('！')
    print("="*80)
    print(f": {total_tasks}")
    print(f": {total_backbones}")
    print(f"outputsequence: {total_sequences}")
    print(f"outputfile: {output_file_path}")
    print('\n:')
    print(' --remove_slash sequence')
    print(' --keep_part binder binder(default,forAF3)')
    print(' --keep_part both receptor+binder(remove)')
    print(' --keep_part receptor receptor')
    print(' --receptor-first sequenceas「receptorin, binderin」argument')
    print(' --auto-detect-order auto「=receptor, =binder」(and --remove_slash )')
    print(' --rename sequence()')
    print(' --name_map savefile(and--rename)')

if __name__ == '__main__':
    main()

