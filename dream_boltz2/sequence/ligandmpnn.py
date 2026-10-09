"""LigandMPNN runner. model_type selects ligand_mpnn, soluble_mpnn, or protein_mpnn weights."""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
from types import SimpleNamespace

import torch
import numpy as np



def _resolve_ligandmpnn_path() -> str:
    """Directory that contains LigandMPNN run.py and model_params/."""
    env = os.environ.get("LIGANDMPNN_DIR")
    if env:
        return env
    project_root = Path(__file__).resolve().parents[2]
    bundled = project_root / "third_party" / "LigandMPNN"
    if (bundled / "run.py").is_file():
        return str(bundled)
    return str(project_root / "external" / "LigandMPNN")


ligandmpnn_path = _resolve_ligandmpnn_path()

if ligandmpnn_path not in sys.path:
    sys.path.insert(0, ligandmpnn_path)


try:
    from run import main as ligandmpnn_main
except ImportError as e:
    error_msg = str(e)
    print(f"[WARNING] LigandMPNN import failed: {error_msg}")
    print(f"Looked in: {ligandmpnn_path}")
    if "prody" in error_msg.lower():
        print("Install prody: pip install prody")
    elif not os.path.exists(ligandmpnn_path):
        print("The bundled copy should be at third_party/LigandMPNN.")
    
    ligandmpnn_main = None


def load_bias_AA_from_json(json_path: str) -> str:
    """load bias AA from json."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        
        
        if "global" in data:
            bias_dict = data["global"]
        else:
            
            bias_dict = {k: v for k, v in data.items() if not k.startswith('_')}
        
        
        bias_items = [f"{aa}:{bias}" for aa, bias in bias_dict.items() if isinstance(bias, (int, float))]
        bias_str = ",".join(bias_items)
        
        return bias_str
    except FileNotFoundError:
        raise FileNotFoundError(f"JSONfilenotto: {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONfileerror: {json_path}, error: {e}")
    except Exception as e:
        raise RuntimeError(f"loadJSONfilefailed: {json_path}, error: {e}")


class LigandMPNNRunner:
    """LigandMPNNRunner."""
    
    def __init__(
        self,
        model_type: str = "auto",  # "auto", "ligand_mpnn", "soluble_mpnn", "protein_mpnn"
        device: Optional[str] = None,
    ):
        """  init  ."""
        if ligandmpnn_main is None:
            raise RuntimeError(
                'LigandMPNN notorfailed.\n'
                'check:\n'
                '1. whetheralready prody: pip install prody or conda install -c conda-forge prody\n'
                '2. LigandMPNN whetherto: ' + ligandmpnn_path + "\n"
                '3. allwhetheralready(torch, numpy, prody)'
            )
        
        self.model_type = model_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ligandmpnn_path = ligandmpnn_path
        
        
        self.model_params_dir = os.path.join(ligandmpnn_path, 'model_params')
        self._check_model_params()
    
    def _check_model_params(self):
        """ check model params."""
        required_models = {
            "ligand_mpnn": "ligandmpnn_v_32_020_25.pt",
            "soluble_mpnn": "solublempnn_v_48_020.pt",
            "protein_mpnn": "proteinmpnn_v_48_020.pt",
        }
        
        for model_name, weight_file in required_models.items():
            weight_path = os.path.join(self.model_params_dir, weight_file)
            if not os.path.exists(weight_path):
                print(f" warning: {model_name} weightfile: {weight_file}")
    
    def _replace_unk_with_ala(self, pdb_path: str, out_dir: str) -> str:
        """ replace unk with ala."""
        pdb_name = Path(pdb_path).stem
        output_path = os.path.join(out_dir, f"{pdb_name}_processed.pdb")
        
        
        with open(pdb_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        
        unk_count = 0
        processed_lines = []
        
        for line in lines:
            
            if line.startswith('ATOM') or line.startswith('HETATM'):
                
                if len(line) > 20 and 'UNK' in line[17:20]:
                    
                    
                    
                    line = line[:17] + 'ALA' + line[20:]
                    unk_count += 1
            
            processed_lines.append(line)
        
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(processed_lines)
        
        if unk_count > 0:
            print(f" already {unk_count} UNKresidueasALA")
            print(f" PDB: {output_path}")
        else:
            print(f"ℹ PDBfileinnoUNKresidue,no")
        
        return output_path
    
    def _detect_model_type(self, pdb_path: str) -> str:
        """ detect model type."""
        try:
            from prody import parsePDB
            pdb = parsePDB(pdb_path)
            
            
            has_ligand = pdb.select('not protein and not water') is not None
            
            if has_ligand:
                return "ligand_mpnn"
            else:
                return "soluble_mpnn"
        except Exception as e:
            print(f" autofailed: {e},defaultuse soluble_mpnn")
            return "soluble_mpnn"
    
    def generate_sequences(
        self,
        pdb_path: str,
        binder_chain: str = "A",
        num_sequences: int = 100,
        temperature: float = 0.1,
        fixed_residues: Optional[str] = None,
        out_dir: Optional[str] = None,
        batch_size: int = 16,
        seed: Optional[int] = None,  
        bias_AA: Optional[Union[str, Dict[str, float]]] = None,  
        bias_AA_json: Optional[str] = None,  
        bias_AA_per_residue: Optional[str] = None,  
        return_pdb_paths: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """generate sequences."""
        
        if self.model_type == "auto":
            model_type = self._detect_model_type(pdb_path)
        else:
            model_type = self.model_type
        
        
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(pdb_path), f"ligandmpnn_output_{os.getpid()}")
        os.makedirs(out_dir, exist_ok=True)
        
        
        pdb_path_processed = self._replace_unk_with_ala(pdb_path, out_dir)
        
        
        args = SimpleNamespace()
        args.model_type = model_type
        
        
        args.seed = seed
        args.pdb_path = pdb_path_processed  
        args.pdb_path_multi = ""  
        args.out_folder = out_dir
        args.chains_to_design = binder_chain
        args.fixed_residues = fixed_residues if fixed_residues else ""
        args.fixed_residues_multi = ""  
        args.redesigned_residues = ""
        args.redesigned_residues_multi = ""  
        
        if batch_size is None:
            batch_size = 16  
        args.batch_size = batch_size
        args.number_of_batches = int(np.ceil(num_sequences / batch_size))
        args.temperature = temperature
        args.omit_AA = "X"  
        args.save_stats = 0
        args.verbose = 1  
        
        
        args.ligand_mpnn_use_side_chain_context = 1 if model_type == "ligand_mpnn" else 0
        
        
        args.checkpoint_protein_mpnn = os.path.join(self.model_params_dir, "proteinmpnn_v_48_020.pt")
        args.checkpoint_Cas9_mpnn = os.path.join(self.model_params_dir, "Cas9mpnn_v_48_020.pt")
        args.checkpoint_Luc7_mpnn = os.path.join(self.model_params_dir, "Luc7mpnn_v_48_020.pt")
        args.checkpoint_per_residue_label_membrane_mpnn = os.path.join(self.model_params_dir, "per_residue_label_membrane_mpnn_v_48_020.pt")
        args.checkpoint_global_label_membrane_mpnn = os.path.join(self.model_params_dir, "global_label_membrane_mpnn_v_48_020.pt")
        
        if model_type == "ligand_mpnn":
            args.checkpoint_ligand_mpnn = os.path.join(self.model_params_dir, "ligandmpnn_v_32_020_25.pt")
        elif model_type == "soluble_mpnn":
            args.checkpoint_soluble_mpnn = os.path.join(self.model_params_dir, "solublempnn_v_48_020.pt")
        elif model_type == "protein_mpnn":
            args.checkpoint_protein_mpnn = os.path.join(self.model_params_dir, "proteinmpnn_v_48_020.pt")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        
        bias_AA_str = ""
        
        
        if bias_AA_json:
            bias_AA_str = load_bias_AA_from_json(bias_AA_json)
        elif bias_AA:
            
            if isinstance(bias_AA, dict):
                
                bias_items = [f"{aa}:{bias}" for aa, bias in bias_AA.items()]
                bias_AA_str = ",".join(bias_items)
            elif isinstance(bias_AA, str):
                
                if os.path.exists(bias_AA) and bias_AA.endswith('.json'):
                    
                    bias_AA_str = load_bias_AA_from_json(bias_AA)
                else:
                    
                    bias_AA_str = bias_AA
        
        args.bias_AA = bias_AA_str
        args.bias_AA_per_residue = bias_AA_per_residue if bias_AA_per_residue else ""  
        args.bias_AA_per_residue_multi = ""  
        args.omit_AA_per_residue = ""
        args.omit_AA_per_residue_multi = ""  
        args.symmetry_residues = ""
        args.symmetry_weights = ""
        args.homo_oligomer = 0
        args.global_transmembrane_label = 0
        args.parse_these_chains_only = ""  
        args.parse_atoms_with_zero_occupancy = 0  
        args.transmembrane_buried = ""  
        args.transmembrane_interface = ""  
        args.ligand_mpnn_use_atom_context = 1 if model_type == "ligand_mpnn" else 0  
        args.ligand_mpnn_cutoff_for_score = 8.0  
        args.fasta_seq_separation = ":"  
        args.file_ending = ""  
        
        
        
        
        
        
        args.zero_indexed = 0  
        args.verbose = 1  
        
        
        print(f"\n LigandMPNN ({model_type})...")
        print(f"  - PDB: {pdb_path}")
        print(f" - chain: {binder_chain}")
        print(f" - sequence: {num_sequences}")
        print(f" -: {temperature}")
        if fixed_residues:
            print(f" - fixedresidue: {fixed_residues}")
        if bias_AA_str:
            print(f" -: {bias_AA_str}")
        if bias_AA_per_residue:
            print(f" - eachresiduefile: {bias_AA_per_residue}")
        
        try:
            ligandmpnn_main(args)
        except Exception as e:
            print(f" LigandMPNN failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        
        print(f"\n checkoutput directory: {out_dir}")
        if os.path.exists(out_dir):
            subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
            print(f" directory: {subdirs}")
            
            seqs_dir = os.path.join(out_dir, 'seqs')
            if os.path.exists(seqs_dir):
                fa_files = [f for f in os.listdir(seqs_dir) if f.endswith('.fa')]
                print(f" FASTAfile: {fa_files}")
            
            backbones_dir = os.path.join(out_dir, 'backbones')
            if os.path.exists(backbones_dir):
                pdb_files = [f for f in os.listdir(backbones_dir) if f.endswith('.pdb')]
                print(f" PDBfile: {len(pdb_files)}")
                if pdb_files:
                    print(f" 5PDBfile: {pdb_files[:5]}")
        else:
            print(f" output directorynotexists: {out_dir}")
        
        
        
        
        pdb_name_processed = Path(pdb_path_processed).stem
        pdb_name_original = Path(pdb_path).stem
        
        
        fasta_path = os.path.join(out_dir, 'seqs', f'{pdb_name_processed}.fa')
        if not os.path.exists(fasta_path):
            
            fasta_path_alt = os.path.join(out_dir, 'seqs', f'{pdb_name_original}.fa')
            if os.path.exists(fasta_path_alt):
                fasta_path = fasta_path_alt
                print(f"ℹ usePDBoutputfile: {fasta_path_alt}")
            else:
                
                seqs_dir = os.path.join(out_dir, 'seqs')
                if os.path.exists(seqs_dir):
                    all_fa_files = [f for f in os.listdir(seqs_dir) if f.endswith('.fa')]
                    if all_fa_files:
                        print(f" outputfilenotto: {fasta_path}")
                        print(f" toFASTAfile: {all_fa_files}")
                        print(f" useafile: {all_fa_files[0]}")
                        fasta_path = os.path.join(seqs_dir, all_fa_files[0])
                    else:
                        print(f" outputfilenotto: {fasta_path}")
                        print(f" seqsdirectoryexistsasempty: {seqs_dir}")
                        return []
                else:
                    print(f" outputfilenotto: {fasta_path}")
                    print(f" seqsdirectorynotexists: {seqs_dir}")
                    return []
        
        sequences = self._parse_fasta(fasta_path)
        print(f" generate {len(sequences)} sequence")
        
        
        
        
        
        
        # 
        
        if return_pdb_paths:
            backbone_dir = os.path.join(out_dir, 'backbones')
            pdb_paths = []
            if os.path.exists(backbone_dir):
                
                
                
                pdb_name = Path(pdb_path_processed).stem
                for i in range(len(sequences)):
                    seq_id = i + 1  
                    pdb_file = os.path.join(backbone_dir, f'{pdb_name}_{seq_id}.pdb')
                    if os.path.exists(pdb_file):
                        pdb_paths.append(pdb_file)
                    else:
                        
                        alt_pdb_file = os.path.join(backbone_dir, f'{pdb_name}_{i}.pdb')
                        if os.path.exists(alt_pdb_file):
                            pdb_paths.append(alt_pdb_file)
                        else:
                            pdb_paths.append(None)  
                
                if any(pdb_paths):
                    print(f" to {sum(1 for p in pdb_paths if p)} PDBfile")
                    print(f" note:thisPDB+sequence,notmatch,notuse")
                    print(f":useBoltz2/AlphaFold2sequence")
                else:
                    print(f" nottoPDBfile(in {backbone_dir})")
            
            return sequences, pdb_paths
        else:
            return sequences
    
    def _parse_fasta(self, fasta_path: str) -> List[str]:
        """ parse fasta."""
        sequences = []
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        
        
        for i in range(2, len(lines)):  
            line = lines[i].strip()
            if line.startswith('>'):
                
                if i + 1 < len(lines):
                    seq = lines[i + 1].strip()
                    if seq:  
                        sequences.append(seq)
        
        return sequences
    
    def generate_sequences_with_confidence(
        self,
        pdb_path: str,
        binder_chain: str = "A",
        num_sequences: int = 100,
        temperature: float = 0.1,
        fixed_residues: Optional[str] = None,
        out_dir: Optional[str] = None,
        bias_AA: Optional[Union[str, Dict[str, float]]] = None,  
        bias_AA_json: Optional[str] = None,  
        bias_AA_per_residue: Optional[str] = None,  
    ) -> List[Tuple[str, float, float]]:
        """generate sequences with confidence."""
        
        sequences = self.generate_sequences(
            pdb_path=pdb_path,
            binder_chain=binder_chain,
            num_sequences=num_sequences,
            temperature=temperature,
            fixed_residues=fixed_residues,
            out_dir=out_dir,
            bias_AA=bias_AA,
            bias_AA_json=bias_AA_json,
            bias_AA_per_residue=bias_AA_per_residue,
        )
        
        
        pdb_name = Path(pdb_path).stem
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(pdb_path), f"ligandmpnn_output_{os.getpid()}")
        fasta_path = os.path.join(out_dir, 'seqs', f'{pdb_name}.fa')
        
        return self._parse_fasta_with_confidence(fasta_path)
    
    def _parse_fasta_with_confidence(
        self, fasta_path: str
    ) -> List[Tuple[str, float, float]]:
        """ parse fasta with confidence."""
        results = []
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        current_confidence = None
        current_ligand_confidence = None
        
        for i, line in enumerate(lines[2:]):  
            if line.startswith('>'):
                
                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        overall_confidence = float(parts[4].split('=')[1])
                        ligand_confidence = float(parts[5].split('=')[1])
                        current_confidence = overall_confidence
                        current_ligand_confidence = ligand_confidence
                    except:
                        current_confidence = 0.0
                        current_ligand_confidence = 0.0
            elif current_confidence is not None:
                
                seq = line.strip()
                if seq:
                    results.append((seq, current_confidence, current_ligand_confidence))
                    current_confidence = None
                    current_ligand_confidence = None
        
        return results


if __name__ == "__main__":
    
    print("=" * 60)
    print("LigandMPNN Runner Test")
    print("=" * 60)
    
    
    runner = LigandMPNNRunner()
    print(f"\n LigandMPNN Runner initok")
    print(f"  - Device: {runner.device}")
    print(f"  - Model params dir: {runner.model_params_dir}")
    
    
    
    print("\n" + "=" * 60)
    print('！')
    print("=" * 60)

