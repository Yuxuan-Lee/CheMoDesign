#!/usr/bin/env python3
"""Convert an a3m MSA to Boltz JSON."""

import json
import sys
import os
from pathlib import Path


def _parse_a3m_entries(a3m_content: str):
    """ parse a3m entries."""
    lines = a3m_content.strip().split('\n')
    entries = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            header = lines[i]
            seq_parts = []
            i += 1
            while i < len(lines) and not lines[i].startswith('>'):
                seq_parts.append(lines[i].strip())
                i += 1
            entries.append((header, ''.join(seq_parts)))
        else:
            i += 1
    return entries


def convert_a3m_to_af3_json(
    a3m_path: str,
    output_path: str = None,
    name: str = None,
    chain_id: str = "A",
    model_seeds: list = None,
    max_msa_seqs: int = None,
):
    """convert a3m to af3 json."""
    
    with open(a3m_path, 'r', encoding='utf-8') as f:
        a3m_content = f.read()
    
    entries = _parse_a3m_entries(a3m_content)
    if not entries:
        print(f"error:nofrom {a3m_path} inparsesequence")
        return False
    
    
    query_seq = entries[0][1].replace('-', '')
    if not query_seq:
        print(f"error:nofrom {a3m_path} inextract query sequence")
        return False
    
    
    if max_msa_seqs is not None and len(entries) > max_msa_seqs:
        entries = entries[:max_msa_seqs]
        a3m_content = '\n'.join(h + '\n' + s for h, s in entries)
    else:
        a3m_content = a3m_content.strip()
    
    msa_count = len(entries)
    
    
    if output_path is None:
        output_path = str(Path(a3m_path).with_suffix('.json'))
    
    if name is None:
        name = Path(a3m_path).stem.replace('_data', '')
    
    if model_seeds is None:
        model_seeds = [1, 22, 333, 4444, 66666]
    
    
    data = {
        "dialect": "alphafold3",
        "version": 3,
        "name": name,
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": query_seq,
                    "modifications": [],
                    "unpairedMsa": a3m_content,
                    "pairedMsa": a3m_content,
                    "templates": []
                }
            }
        ],
        "modelSeeds": model_seeds,
        "bondedAtomPairs": None,
        "userCCD": None
    }
    
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] convertok！")
    print(f" input: {a3m_path}")
    print(f" output: {output_path}")
    print(f": {name}")
    print(f" chainID: {chain_id}")
    print(f" Querysequencelength: {len(query_seq)}")
    print(f" MSAsequence: {msa_count}" + (f" (alreadyas {max_msa_seqs} )" if max_msa_seqs else ""))
    print(f": {model_seeds}")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description=' A3M MSA convertas AlphaFold3 JSON '
    )
    parser.add_argument('a3m_path', help='input A3M file path')
    parser.add_argument('--output', '-o', default=None, help='output JSON (default.json)')
    parser.add_argument('--name', '-n', default=None, help='(defaultfromfileextract)')
    parser.add_argument('--chain_id', '-c', default='A', help='chain ID(default A)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 22, 333, 4444, 66666],
                       help='list(default: 1 22 333 4444 66666)')
    parser.add_argument('--max_seqs', '-m', type=int, default=None,
                       help=' MSA sequence( query), N , 5000;not')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.a3m_path):
        print(f"error:filenotexists: {args.a3m_path}")
        sys.exit(1)
    
    success = convert_a3m_to_af3_json(
        a3m_path=args.a3m_path,
        output_path=args.output,
        name=args.name,
        chain_id=args.chain_id,
        model_seeds=args.seeds,
        max_msa_seqs=args.max_seqs,
    )
    
    if not success:
        sys.exit(1)

