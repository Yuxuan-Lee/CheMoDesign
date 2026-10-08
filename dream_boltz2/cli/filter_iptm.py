#!/usr/bin/env python3
"""Filter prediction summaries by ipTM."""

import pandas as pd
import shutil
from pathlib import Path
import argparse
import sys


def filter_high_iptm_yamls(
    labels_csv: str,
    yaml_inputs_dir: str,
    output_dir: str,
    iptm_threshold: float = 0.7,
    verbose: bool = True,
):
    """filter high iptm yamls."""
    
    if verbose:
        print(f"[INFO] read labels.csv: {labels_csv}")
    df = pd.read_csv(labels_csv)
    
    
    high_iptm_df = df[df['iptm'] > iptm_threshold].copy()
    
    if len(high_iptm_df) == 0:
        print(f"[WARN] noto iPTM > {iptm_threshold} ")
        return
    
    if verbose:
        print(f"[OK] to {len(high_iptm_df)} iPTM > {iptm_threshold} ")
        print(f" iPTM: [{high_iptm_df['iptm'].min():.4f}, {high_iptm_df['iptm'].max():.4f}]")
        print(f" iPTM: {high_iptm_df['iptm'].mean():.4f}")
    
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    yaml_inputs_path = Path(yaml_inputs_dir)
    
    
    copied_count = 0
    missing_count = 0
    copied_files = []
    
    
    all_yaml_files = list(yaml_inputs_path.glob("seq_*.yaml"))
    yaml_file_map = {}  
    for yaml_file in all_yaml_files:
        
        base_name = yaml_file.stem  
        
        
        if base_name.startswith("seq_"):
            
            num_part = base_name[4:]  
            
            if len(num_part) >= 3:
                
                possible_base = f"seq_{num_part[:-2]}"
                if possible_base not in yaml_file_map:
                    yaml_file_map[possible_base] = yaml_file
            
            if base_name not in yaml_file_map:
                yaml_file_map[base_name] = yaml_file
    
    for idx, row in high_iptm_df.iterrows():
        seq_index = row['index']
        base_name = f"seq_{seq_index}"
        
        
        yaml_filename = f"{base_name}.yaml"
        source_yaml = yaml_inputs_path / yaml_filename
        
        
        
        if not source_yaml.exists():
            matching_files = list(yaml_inputs_path.glob(f"{base_name}*.yaml"))
            if matching_files:
                
                source_yaml = matching_files[0]
                yaml_filename = source_yaml.name
            else:
                missing_count += 1
                if verbose:
                    print(f"[WARN] filenotexists: seq_{seq_index}.yaml (matchfailed)")
                continue
        
        
        target_yaml = output_path / yaml_filename
        try:
            shutil.copy2(source_yaml, target_yaml)
            copied_count += 1
            copied_files.append({
                'index': seq_index,
                'iptm': row['iptm'],
                'evolution_score': row.get('evolution_score', 'N/A'),
                'filename': yaml_filename,
            })
        except Exception as e:
            missing_count += 1
            if verbose:
                print(f"[ERROR] filefailed {source_yaml}: {e}")
    
    
    print(f"\n{'='*70}")
    print(f"[INFO] screenresult")
    print(f"{'='*70}")
    print(f"[OK] ok: {copied_count} YAML file")
    if missing_count > 0:
        print(f"[WARN] file: {missing_count} ")
    print(f"[DIR] output directory: {output_path}")
    
    
    if len(copied_files) > 0:
        summary_df = pd.DataFrame(copied_files)
        print(f"\n[TOP] Top-10 iPTM:")
        top_10 = summary_df.nlargest(10, 'iptm')
        for i, row in top_10.iterrows():
            evolution_score = row.get('evolution_score', 'N/A')
            if evolution_score != 'N/A':
                print(f"   {i+1}. seq_{row['index']}: iPTM={row['iptm']:.4f}, score={evolution_score:.4f}")
            else:
                print(f"   {i+1}. seq_{row['index']}: iPTM={row['iptm']:.4f}")
    else:
        print(f"[WARN] nookfile")


def main():
    parser = argparse.ArgumentParser(
        description='screen iPTM > 0.7 YAML file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n:\n # usedefaultthreshold 0.7\n python filter_high_iptm_yamls.py \\\n --labels_csv inverse_design_v2/boltz2_inputs/labels.csv \\\n --yaml_inputs_dir inverse_design_v2/boltz2_inputs/yaml_inputs \\\n --output_dir inverse_design_v2/boltz2_inputs/high_iptm_yamls\n \n # threshold 0.8\n python filter_high_iptm_yamls.py \\\n --labels_csv inverse_design_v2/boltz2_inputs/labels.csv \\\n --yaml_inputs_dir inverse_design_v2/boltz2_inputs/yaml_inputs \\\n --output_dir inverse_design_v2/boltz2_inputs/high_iptm_yamls \\\n --iptm_threshold 0.8\n '
    )
    
    parser.add_argument(
        '--labels_csv',
        type=str,
        required=True,
        help='labels.csv file path'
    )
    parser.add_argument(
        '--yaml_inputs_dir',
        type=str,
        required=True,
        help='YAML fileinputdirectory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='output directory( YAML filetothis)'
    )
    parser.add_argument(
        '--iptm_threshold',
        type=float,
        default=0.7,
        help='iPTM threshold(default 0.7)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='mode(notinfo)'
    )
    
    args = parser.parse_args()
    
    filter_high_iptm_yamls(
        labels_csv=args.labels_csv,
        yaml_inputs_dir=args.yaml_inputs_dir,
        output_dir=args.output_dir,
        iptm_threshold=args.iptm_threshold,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()

