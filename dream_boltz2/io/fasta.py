"""FASTA reader."""

from typing import List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def read_fasta(fasta_path: str) -> List[str]:
    """read fasta."""
    fasta_path = Path(fasta_path)
    
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTAfilenotexists: {fasta_path}")
    
    sequences = []
    with open(fasta_path, 'r', encoding='utf-8') as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                
                if line:
                    current_seq += line
        
        
        if current_seq:
            sequences.append(current_seq)
    
    logger.info(f"from {fasta_path} readto {len(sequences)} sequence")
    
    return sequences


if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) < 2:
        print(': python read_fasta.py <fasta_file>')
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    sequences = read_fasta(fasta_file)
    
    print(f"\n readok！")
    print(f" file: {fasta_file}")
    print(f" sequence: {len(sequences)}")
    
    if len(sequences) > 0:
        print(f"\n5sequencelength:")
        for i, seq in enumerate(sequences[:5], 1):
            print(f" sequence {i}: {len(seq)} residue")
            if len(seq) <= 60:
                print(f" sequence: {seq}")
            else:
                print(f" sequence: {seq[:30]}...{seq[-30:]}")

