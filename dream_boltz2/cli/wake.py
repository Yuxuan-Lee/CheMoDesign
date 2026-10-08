#!/usr/bin/env python3
"""Waking: sequence-conditioned truncated diffusion, including deferred covalent ligands."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _nanobody_bfactor_cdr_unreliable(
    regions: Optional[List[Tuple[int, int]]],
    binder_len: int,
) -> bool:
    """ nanobody bfactor cdr unreliable."""
    if binder_len <= 0:
        return True
    if not regions:
        return True
    if len(regions) >= 2:
        return False
    s, e = regions[0]
    
    if s == 0 and (e - s) >= max(4, int(0.85 * binder_len)):
        return True
    return False


def build_start_coords_from_pdb(
    pdb_path: str,
    feats: Dict[str, torch.Tensor],
    receptor_chain_id: str = "A",
    binder_chain_id: str = "B",
    device: str = "cuda",
) -> torch.Tensor:
    """build start coords from pdb."""
    from Bio.PDB import PDBParser
    import numpy as np

    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]

    ca_by_chain: Dict[str, List[np.ndarray]] = {}  # chain_id -> [CA_coords per residue]
    for chain in model:
        cid = chain.id
        cas = []
        for res in chain.get_residues():
            
            hetflag = res.id[0]
            if hetflag == "W":  
                continue
            if "CA" in res:
                cas.append(res["CA"].get_vector().get_array())
        if cas:
            ca_by_chain[cid] = cas

    
    
    atom_to_token = feats["atom_to_token"].cpu()
    if atom_to_token.dim() == 3:
        atom_to_token = atom_to_token[0]  # [N_atoms, N_tokens]
    atom_to_token_idx = atom_to_token.argmax(dim=-1)  # [N_atoms] -> token index

    # token_to_center_atom: [1, N_tokens, N_atoms] one-hot
    token_to_center = feats["token_to_center_atom"].cpu()
    if token_to_center.dim() == 3:
        token_to_center = token_to_center[0]  # [N_tokens, N_atoms]
    center_atom_idx = token_to_center.argmax(dim=-1)  # [N_tokens] -> atom index of CA

    N_atoms = atom_to_token.shape[0]
    N_tokens = atom_to_token.shape[1]

    # atom_pad_mask: [1, N_atoms]
    atom_mask = feats["atom_pad_mask"].cpu()
    if atom_mask.dim() == 2:
        atom_mask = atom_mask[0]  # [N_atoms]

    
    
    asym_id = feats.get("asym_id", None)
    if asym_id is not None:
        if asym_id.dim() >= 2:
            asym_id = asym_id[0]  # [N_tokens]
        asym_id = asym_id.cpu()  

    # token_pad_mask: [1, N_tokens]
    token_mask = feats.get("token_pad_mask", None)
    if token_mask is not None:
        if token_mask.dim() >= 2:
            token_mask = token_mask[0]
        token_mask = token_mask.cpu()

    
    
    
    unique_asym = asym_id.unique(sorted=True) if asym_id is not None else torch.tensor([0])

    
    token_ca = torch.zeros(N_tokens, 3)

    
    
    
    chain_order = []
    
    
    
    for cid in [receptor_chain_id, binder_chain_id]:
        if cid in ca_by_chain:
            chain_order.append(cid)

    for asym_idx, asym_val in enumerate(unique_asym):
        if asym_idx >= len(chain_order):
            break
        cid = chain_order[asym_idx]
        cas = ca_by_chain.get(cid, [])

        
        tok_mask = (asym_id == asym_val)
        if token_mask is not None:
            tok_mask = tok_mask & (token_mask > 0)
        tok_indices = torch.where(tok_mask)[0]

        for ti, tok_idx in enumerate(tok_indices):
            if ti < len(cas):
                token_ca[tok_idx] = torch.from_numpy(cas[ti]).float()
            else:
                
                if cas:
                    token_ca[tok_idx] = torch.from_numpy(cas[-1]).float() + torch.randn(3) * 2.0
                

    
    
    spread = 2.0  

    
    start_coords = token_ca[atom_to_token_idx]  # [N_atoms, 3]

    
    scatter_noise = spread * torch.randn(N_atoms, 3)

    
    
    is_ca = torch.zeros(N_atoms, dtype=torch.bool)
    for tok in range(N_tokens):
        ca_atom = center_atom_idx[tok].item()
        if 0 <= ca_atom < N_atoms:
            is_ca[ca_atom] = True
    scatter_noise[is_ca] = 0.0

    
    pad_mask = atom_mask < 0.5
    scatter_noise[pad_mask] = 0.0

    start_coords = start_coords + scatter_noise

    
    
    start_coords = start_coords.unsqueeze(0).to(device)

    
    valid_mask = atom_mask > 0.5
    valid_coords = start_coords[0][valid_mask]
    print(f" [coordinates] N_atoms={N_atoms}, ={valid_mask.sum().item()}, "
          f"coordinates=[{valid_coords.min():.1f}, {valid_coords.max():.1f}], "
          f"std={valid_coords.std():.1f}Å (no,diffusion)")

    return start_coords


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML required dict: {path}")
    return data


def _extract_binder_chain_ids(project_yaml: Dict[str, Any]) -> List[str]:
    dream = project_yaml.get("dream", {})
    if not isinstance(dream, dict):
        raise ValueError('YAML dream required dict')
    binder = dream.get("binder_chains", [])
    if isinstance(binder, str):
        binder = [binder]
    if not isinstance(binder, list) or not binder:
        raise ValueError('YAML required dream.binder_chains(empty)')
    return [str(x) for x in binder]


def _extract_receptor_chain_ids(project_yaml: Dict[str, Any]) -> List[str]:
    dream = project_yaml.get("dream", {})
    if not isinstance(dream, dict):
        raise ValueError('YAML dream required dict')
    receptor = dream.get("receptor_chains", [])
    if isinstance(receptor, str):
        receptor = [receptor]
    if not isinstance(receptor, list) or not receptor:
        raise ValueError('YAML required dream.receptor_chains(empty)')
    return [str(x) for x in receptor]


def _extract_ligands_from_yaml(project_yaml: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ extract ligands from yaml."""
    ligands = []
    sequences = project_yaml.get("sequences", [])
    if isinstance(sequences, list):
        for item in sequences:
            if not isinstance(item, dict):
                continue
            if "ligand" in item:
                lig = item["ligand"]
                if isinstance(lig, dict):
                    lig_dict = {}
                    if "id" in lig:
                        lig_dict["id"] = lig["id"]
                    if "ccd" in lig:
                        lig_dict["ccd"] = lig["ccd"]
                    if "smiles" in lig:
                        lig_dict["smiles"] = lig["smiles"]
                    if lig_dict:
                        ligands.append(lig_dict)
    return ligands


def _extract_constraints_from_yaml(project_yaml: Dict[str, Any]) -> List:
    """ extract constraints from yaml."""
    from dream_boltz2.config import parse_constraint
    constraints = project_yaml.get("constraints", [])
    if not isinstance(constraints, list):
        return []
    return [parse_constraint(c) for c in constraints if isinstance(c, dict)]


def _get_receptor_sequence_from_yaml(project_yaml: Dict[str, Any], receptor_chain_ids: List[str]) -> Optional[str]:
    """ get receptor sequence from yaml."""
    sequences = project_yaml.get("sequences", [])
    if not isinstance(sequences, list):
        return None
    for item in sequences:
        if not isinstance(item, dict):
            continue
        if "protein" in item:
            protein = item["protein"]
            if isinstance(protein, dict):
                chain_id = protein.get("id")
                if chain_id in receptor_chain_ids:
                    seq = protein.get("sequence")
                    if isinstance(seq, str):
                        return seq
    return None


def _get_receptor_msa_from_yaml(
    project_yaml: Dict[str, Any],
    receptor_chain_ids: List[str],
    yaml_path: Path,
) -> Optional[str]:
    """Receptor MSA path, resolved relative to the YAML file when that file exists."""
    from dream_boltz2.config import resolve_msa_path

    sequences = project_yaml.get("sequences", [])
    if not isinstance(sequences, list):
        return None
    for item in sequences:
        if not isinstance(item, dict):
            continue
        if "protein" in item:
            protein = item["protein"]
            if isinstance(protein, dict):
                chain_id = protein.get("id")
                if chain_id in receptor_chain_ids:
                    msa = protein.get("msa")
                    if isinstance(msa, str):
                        return resolve_msa_path(msa, yaml_path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backbone relaxation (forward-only) with mandatory binder template CIF constraint."
    )
    parser.add_argument(
        "--yaml_config",
        required=True,
        help="Project YAML (for example examples/pd_l1.yaml).",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help='backbone PDB inputdirectory(defaultfrom YAML dream.output.dir screen_results)',
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help='output directory(default: <dream.output.dir>/relax_results )',
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help='Boltz2 checkpoint ()',
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='(default cuda)',
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help='eachinputbackboneoutput(default5)',
    )
    
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=None,
        help='(deprecated,use --num_samples)eachinputbackboneoutput',
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=25,
        help='diffusionsamplingstep(default25)',
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=None,
        help='(deprecated,notuse recycling)',
    )
    parser.add_argument(
        "--template_threshold",
        type=float,
        default=3.0,
        help='binder template constraintthreshold(Å,default3.0)',
    )
    parser.add_argument(
        "--num_fake_msa",
        type=int,
        default=4000,
        help='fake paired MSA sequence(default4000).interface,',
    )
    
    parser.add_argument(
        "--mpnn_num_seqs",
        type=int,
        default=10,
        help='LigandMPNN generatesequence(default10,top1)',
    )
    parser.add_argument(
        "--mpnn_temperature",
        type=float,
        default=0.2,
        help='LigandMPNN sampling(default0.2)',
    )
    parser.add_argument(
        "--mpnn_bias_AA_json",
        default=None,
        help='JSONfile path(optional, batch_design_sequences.py)',
    )
    parser.add_argument(
        "--skip_mpnn",
        action="store_true",
        help='skip LigandMPNN sequence(,to poly-A)',
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help=':frombackbone PDB coordinates(CA + chain), sigma .'
             'allfrom(notnoise -> not),interface.',
    )
    parser.add_argument(
        "--warm_start_sigma",
        type=float,
        default=20.0,
        help=' sigma(default20.0). sigma tothis,noisestep.'
             'sigma=5->0.99Å(not);sigma=10->2.4Å;sigma=15->4.7Å;'
             'sigma=20->4.5Å/2.8Å(,backbone).',
    )
    parser.add_argument(
        "--nanobody_mode",
        action="store_true",
        help='Nanobodymode:frominputPDBB-factorautoCDRregion(B<0.99=CDR, B>=0.99=framework),'
             'inoutputPDBinkeepB-factor(framework=100.0, CDR=0.0),'
             'batch_design_sequences.pyautofixedresidue.',
    )
    parser.add_argument(
        "--auto_convert",
        action="store_true",
        help='auto input_dir in.cif file,sequencevschainIDconvertas relax-ready PDB.'
             'convert PDB in input_dir _converted/ directory,autoinput.'
             'support AF3/BindCraft outputchainID CIF file.',
    )
    parser.add_argument(
        "--identity_threshold",
        type=float,
        default=0.85,
        help='--auto_convert mode,receptor sequencematchthreshold(default0.85)',
    )
    parser.add_argument(
        "--skip_cdr_fix",
        action="store_true",
        help='--auto_convert mode,skip CDR B-factor annotate(defaultvs nanobody autoannotate)',
    )
    parser.add_argument(
        "--covalent_deferred",
        action="store_true",
        help=(
            "Round 2 of deferred covalent waking. Still runs LigandMPNN, reuses the "
            "attachment site (B-factor 100 or attach_sites.jsonl), and relaxes with the bond. "
            "Without this flag, round 1 runs LigandMPNN, picks a CA-CA site, injects the ligand, "
            "mutates the attachment residue to Gly, and relaxes."
        ),
    )

    args = parser.parse_args()
    
    
    if args.diffusion_samples is not None:
        args.num_samples = args.diffusion_samples
        print(f"[WARNING] --diffusion_samples deprecated,use --num_samples.current: {args.num_samples}")

    
    yaml_path = Path(args.yaml_config).resolve()
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    boltz_src = os.environ.get("BOLTZ_SRC")
    if boltz_src and boltz_src not in sys.path:
        sys.path.insert(0, boltz_src)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(project_root) + os.pathsep + child_env.get("PYTHONPATH", "")

    
    from dream_boltz2.io.cif_template import pdb_to_boltz2_template_cif_from_yaml
    from dream_boltz2.features.complex import FeaturePreparatorComplex
    from dream_boltz2.model.diffusion import DiffusionWrapper
    from dream_boltz2.model.pairformer import PairformerWrapper
    from dream_boltz2.io.pdb import write_coords_to_pdb
    from dream_boltz2.msa.fake_msa import prepare_msa_generator_data, generate_dynamic_msa
    from dream_boltz2.msa.merge import merge_receptor_msas_to_csv

    
    from boltz.main import (
        Boltz2DiffusionParams,
        PairformerArgsV2,
        MSAModuleArgs,
        BoltzSteeringParams
    )
    from dataclasses import asdict

    def load_boltz_model(checkpoint_path: str, device: str = 'cuda', use_physical_guidance: bool = False):
        """load boltz model."""
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
        predict_args = {
            "recycling_steps": 3,
            "sampling_steps": 15,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        from boltz.model.models.boltz2 import Boltz2
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

    
    base_yaml = _load_yaml(yaml_path)
    binder_chain_ids = _extract_binder_chain_ids(base_yaml)
    receptor_chain_ids = _extract_receptor_chain_ids(base_yaml)
    ligands = _extract_ligands_from_yaml(base_yaml)
    yaml_constraints = _extract_constraints_from_yaml(base_yaml)
    receptor_sequence = _get_receptor_sequence_from_yaml(base_yaml, receptor_chain_ids)
    receptor_msa_file = _get_receptor_msa_from_yaml(base_yaml, receptor_chain_ids, yaml_path)

    # Deferred covalent: YAML covalent_mode: deferred -> inject during relax.
    # Round 1 (no --covalent_deferred): LigandMPNN -> CA-CA attach -> covalent relax.
    # Round 2 (--covalent_deferred): LigandMPNN again -> reuse attach site -> covalent relax.
    covalent_cfg = None
    attach_site_records: List[Dict[str, Any]] = []
    attach_site_map: Dict[str, Dict[str, Any]] = {}
    from dream_boltz2.covalent.deferred import (
        build_covalent_constraints,
        collect_attach_site_maps,
        detect_attach_site_from_bfactor,
        lookup_attach_res_from_records,
        midpoint_ca,
        mutate_sequence_to_aa,
        parse_covalent_config,
        receptor_attach_chain,
        receptor_attach_res_1based,
        seed_ligand_start_coords,
        select_binder_attach_site,
        write_attach_sites_jsonl,
    )

    covalent_cfg = parse_covalent_config(base_yaml)
    if args.covalent_deferred and not covalent_cfg.enabled:
        raise ValueError(
            '--covalent_deferred need YAML in dream.covalent_mode: deferred '
            ' dream.covalent config'
        )
    if covalent_cfg.enabled:
        ligands = covalent_cfg.ligands()
        # constraints are per-PDB (binder site varies); base list rebuilt each PDB
        yaml_constraints = []
        round_label = (
            'secondround(LigandMPNN -> crosslinksite -> covalent relax)'
            if args.covalent_deferred
            else 'firstround(LigandMPNN -> CA-CA -> covalent relax)'
        )
        print(f"\n{'='*70}")
        print(f" Deferred covalent {round_label}")
        print(f"  ligand: {ligands}")
        print(f"  skip_mpnn: {args.skip_mpnn}")
        print(f"  receptor_bond: {list(covalent_cfg.receptor_bond_atom1)} <-> "
              f"{list(covalent_cfg.receptor_bond_atom2)}")
        print(f"  binder_ligand_atom: {list(covalent_cfg.binder_ligand_atom)}")
        print(f"  mutate_to: {covalent_cfg.mutate_to}")
        print(
            f"  attach_site_method: {covalent_cfg.attach_site_method} "
            f"CA-CA [{covalent_cfg.attach_ca_min:.1f}, {covalent_cfg.attach_ca_max:.1f}]Å"
        )
        print("="*70)
    else:
        covalent_cfg = None

    if receptor_sequence is None:
        raise ValueError('nofrom YAML extract receptor sequence')

    
    dream = base_yaml.get("dream", {}) if isinstance(base_yaml.get("dream", {}), dict) else {}
    if dream.get("design_type") == "nanobody" and not args.nanobody_mode:
        args.nanobody_mode = True
        print(
            '[relax] YAML dream.design_type=nanobody -> auto --nanobody_mode '
            '(relax output PDB annotateframework/CDR B-factor,chain design)'
        )
    output_dir = Path(dream.get("output", {}).get("dir", "./output")).expanduser().resolve()

    input_dir = (output_dir / "screen_results").resolve() if args.input_dir is None else Path(args.input_dir).expanduser().resolve()
    out_root = (output_dir / "relax_results").resolve() if args.out_dir is None else Path(args.out_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir notexists: {input_dir}")

    if covalent_cfg is not None and args.covalent_deferred:
        attach_site_map = collect_attach_site_maps(input_dir)
        if attach_site_map:
            print(f" [covalent] attach_sites.jsonl:{len(attach_site_map)} index")

    # ======================================================================
    
    # ======================================================================
    if args.auto_convert:
        import glob as _glob
        cif_files = sorted(_glob.glob(str(input_dir / "*.cif")))
        if cif_files:
            from dream_boltz2.io.cif_to_pdb import (
                extract_sequences_from_cif,
                sequence_identity,
                identify_chains,
                convert_cif_to_pdb,
            )

            design_type = dream.get("design_type", "binder")
            is_nanobody = design_type == "nanobody"

            converted_dir = input_dir.parent / (input_dir.name + "_converted")
            converted_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*70}")
            print(f"Step -1: CIF -> PDB autoconvert ({len(cif_files)} CIF file)")
            print(f"{'='*70}")
            print(f"  Receptor seq len: {len(receptor_sequence)}")
            print(f"  Identity threshold: {args.identity_threshold}")
            print(f"  Output: {converted_dir}")

            convert_ok = 0
            convert_skip = 0
            for cif_path_str in cif_files:
                cif_name = Path(cif_path_str).name
                out_name = Path(cif_path_str).stem + ".pdb"
                out_pdb = converted_dir / out_name

                try:
                    cif_seqs = extract_sequences_from_cif(cif_path_str)
                except Exception as e:
                    print(f"  [{cif_name}] CIF parsefailed: {e}")
                    convert_skip += 1
                    continue

                if not cif_seqs:
                    print(f"  [{cif_name}] nochain,skip")
                    convert_skip += 1
                    continue

                chain_mapping = identify_chains(
                    cif_seqs, receptor_sequence,
                    receptor_chain_ids, binder_chain_ids,
                    ligands, identity_threshold=args.identity_threshold,
                )
                if chain_mapping is None:
                    best_id = max(
                        (sequence_identity(s, receptor_sequence) for s in cif_seqs.values()),
                        default=0.0,
                    )
                    print(f"  [{cif_name}] no receptor match (best={best_id:.2f}),skip")
                    convert_skip += 1
                    continue

                convert_cif_to_pdb(cif_path_str, chain_mapping, str(out_pdb))

                
                if is_nanobody and not args.skip_cdr_fix:
                    try:
                        from dream_boltz2.sequence.cdr import validate_and_fix_cdr_bfactor
                        is_ok, cdr_regions, template = validate_and_fix_cdr_bfactor(
                            str(out_pdb), binder_chain=binder_chain_ids[0],
                        )
                        cdr_info = f"CDR={template}" if template else "CDR=none"
                    except Exception:
                        cdr_info = "CDR=failed"
                else:
                    cdr_info = "no-CDR"

                
                mapping_str = ", ".join(
                    f"{cid}->{tid}({role})"
                    for cid, (role, tid) in chain_mapping.items()
                )
                print(f"  [{cif_name}] OK: {mapping_str} | {cdr_info} -> {out_name}")
                convert_ok += 1

            print(f"\n convert: {convert_ok} ok, {convert_skip} skip")

            if convert_ok > 0:
                
                input_dir = converted_dir
                print(f" input_dir alreadyto: {input_dir}")
            else:
                print(f" nookconvertfile,use input_dir")
        else:
            print(f"\n[--auto_convert] input_dir inno.cif file,skipconvert")

    # ======================================================================
    
    # ======================================================================
    binder_seq_map: Dict[str, str] = {}  # {pdb_stem: binder_sequence}

    if not args.skip_mpnn:
        mpnn_output_dir = out_root / "mpnn_sequences"
        mpnn_filtered_dir = out_root / "mpnn_filtered"
        binder_chain_id_for_mpnn = binder_chain_ids[0]
        receptor_chain_id_for_mpnn = receptor_chain_ids[0] if receptor_chain_ids else None

        # Step 0a: batch_design_sequences.py
        cmd1 = [
            sys.executable, "-m", "dream_boltz2.cli.sequences",
            "--input_dir", str(input_dir),
            "--binder_chain", binder_chain_id_for_mpnn,
            "--yaml_config", str(yaml_path),
            "--num_sequences", str(args.mpnn_num_seqs),
            "--temperature", str(args.mpnn_temperature),
            "--output_dir", str(mpnn_output_dir),
        ]
        if receptor_chain_id_for_mpnn:
            cmd1 += ["--receptor_chain", receptor_chain_id_for_mpnn]
        if args.mpnn_bias_AA_json:
            cmd1 += ["--bias_AA_json", args.mpnn_bias_AA_json]

        print(f"\n{'='*70}")
        print('Step 0a: LigandMPNN sequence')
        print(f"{'='*70}")
        print(f": {' '.join(cmd1)}")
        ret1 = subprocess.run(cmd1, env=child_env)
        if ret1.returncode != 0:
            raise RuntimeError(f"batch_design_sequences.py failed,: {ret1.returncode}")
        print(f" LigandMPNN sequence: {mpnn_output_dir}")

        
        cmd2 = [
            sys.executable, "-m", "dream_boltz2.cli.filter_designs",
            "--outputs_dir", str(mpnn_output_dir),
            "--output_folder", str(mpnn_filtered_dir),
            "--top_n", "1",
            "--score_type", "overall_confidence",
            "--remove_slash",
            "--keep_part", "binder",
            "--auto-detect-order",
        ]

        print(f"\n{'='*70}")
        print('Step 0b: screeneachbackbone top1 binder sequence')
        print(f"{'='*70}")
        print(f": {' '.join(cmd2)}")
        ret2 = subprocess.run(cmd2, env=child_env)
        if ret2.returncode != 0:
            raise RuntimeError(f"filter_best_designs.py failed,: {ret2.returncode}")
        print(f" sequencescreen: {mpnn_filtered_dir}")

        
        
        input_pdb_stems = {p.stem for p in sorted(input_dir.glob("*.pdb"))}

        def _extract_pdb_stem(header_line: str) -> str:
            """ extract pdb stem."""
            h = header_line.lstrip(">").strip()
            
            if "__" in h:
                candidate = h.split("__")[0]
                if candidate in input_pdb_stems:
                    return candidate
            
            name_part = h.split(",")[0].strip()
            
            if name_part.endswith("_processed"):
                candidate = name_part[: -len("_processed")]
                if candidate in input_pdb_stems:
                    return candidate
            
            if name_part in input_pdb_stems:
                return name_part
            
            best = ""
            for stem in input_pdb_stems:
                if h.startswith(stem) and len(stem) > len(best):
                    best = stem
            return best if best else name_part

        filtered_fasta = mpnn_filtered_dir / "best_designs.fa"
        if filtered_fasta.exists():
            with filtered_fasta.open("r", encoding="utf-8") as f:
                header, seq = "", ""
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if header and seq:
                            pdb_stem_key = _extract_pdb_stem(header)
                            binder_seq_map[pdb_stem_key] = seq
                        header = line
                        seq = ""
                    else:
                        seq += line
                if header and seq:
                    pdb_stem_key = _extract_pdb_stem(header)
                    binder_seq_map[pdb_stem_key] = seq

            print(f"\n parseto {len(binder_seq_map)} backbone binder sequence")
            for stem, s in list(binder_seq_map.items())[:3]:
                print(f"  {stem}: {s[:40]}{'...' if len(s) > 40 else ''}")
        else:
            print(f" screenoutputnotexists: {filtered_fasta},to PDB sequence")
    else:
        print('\n⏭ --skip_mpnn: skip LigandMPNN,use PDB insequence')

    
    if args.checkpoint is None:
        raise ValueError('required --checkpoint argument(Boltz2 checkpoint )')
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint notexists: {checkpoint_path}")

    print(f"load Boltz2: {checkpoint_path}")
    boltz_model = load_boltz_model(str(checkpoint_path), device=args.device, use_physical_guidance=False)

    
    feature_prep = FeaturePreparatorComplex(
        boltz_model=boltz_model,
        device=args.device,
        mode='complex'
    )

    
    pairformer = PairformerWrapper(
        boltz_model=boltz_model,
        num_recycles=3,
        freeze_recycle_layers=True,
        monitor_z=True,
        feature_prep=feature_prep,
        use_checkpointing=False,
        use_interface_mask=False,
    )

    diffusion = DiffusionWrapper(
        boltz_model=boltz_model,
        device=args.device,
    )

    
    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        raise RuntimeError(f"nottoinput PDB: {input_dir}")

    print(f"\nto {len(pdb_files)} inputbackbone,...")

    for pdb_idx, pdb_path in enumerate(pdb_files, 1):
        pdb_stem = pdb_path.stem
        run_dir = out_root / pdb_stem
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{pdb_idx}/{len(pdb_files)}]: {pdb_path.name}")

        
        pdb_cdr_regions = None
        if args.nanobody_mode:
            from dream_boltz2.sequence.cdr import (
                detect_cdr_regions_from_bfactor,
                detect_cdr_regions_from_sequence,
                extract_sequence_from_pdb,
            )
            binder_chain_id_for_detect = binder_chain_ids[0]
            pdb_cdr_regions = detect_cdr_regions_from_bfactor(
                str(pdb_path), binder_chain_id_for_detect
            )
            bseq_for_cdr = extract_sequence_from_pdb(str(pdb_path), binder_chain_id_for_detect)
            n_bind_ca = len(bseq_for_cdr)
            if _nanobody_bfactor_cdr_unreliable(pdb_cdr_regions, n_bind_ca):
                tmpl_name, seq_cdr = detect_cdr_regions_from_sequence(bseq_for_cdr)
                if seq_cdr:
                    pdb_cdr_regions = seq_cdr
                    print(
                        f" CDR:input B-factor noframework/CDR(: B≈0),"
                        f"alreadysequencematchto {len(seq_cdr)} CDR "
                        + (f"(template={tmpl_name})" if tmpl_name else "")
                    )
                else:
                    print(
                        ' B-factor CDR notsequencetemplatenotmatch:'
                        'output PDB binder B-factor as 0(chain design)'
                    )
                    pdb_cdr_regions = None
            elif pdb_cdr_regions:
                print(f" B-factor CDR: to {len(pdb_cdr_regions)} CDRregion")
                for i, (s, e) in enumerate(pdb_cdr_regions, 1):
                    print(f"     CDR{i}: [{s}, {e}) ({e-s}residue)")
            else:
                print(f" B-factor CDR: nottoCDRregion(inputPDBnoB-factor)")

        
        template_cif = run_dir / f"{pdb_stem}.binder_template.cif"
        print(f" generate binder template CIF: {template_cif}")
        try:
            pdb_to_boltz2_template_cif_from_yaml(
                input_pdb=pdb_path,
                project_yaml=yaml_path,
                output_cif=template_cif,
                normalize_output_chain_ids=False,
            )
            print(f" Template CIF generateok")
        except Exception as e:
            print(f" Template CIF generatefailed: {e}")
            import traceback
            traceback.print_exc()
            continue

        
        binder_chain_id = binder_chain_ids[0]  
        receptor_chain_id = receptor_chain_ids[0] if receptor_chain_ids else "A"

        
        
        template_id_for_cif = None  

        print(f" features(binder template: {template_cif}, chain: {binder_chain_id}, force=True, threshold={args.template_threshold})")

        fixed_binder_indices_0based: Optional[List[int]] = None
        covalent_midpoint = None

        try:
            
            from dream_boltz2.io.pdb import parse_pdb_chain
            binder_seq_from_pdb, binder_coords, _ = parse_pdb_chain(str(pdb_path), binder_chain_id, verbose=False)
            binder_length = len(binder_seq_from_pdb) if binder_seq_from_pdb else 100  
            print(f" from PDB extract binder length: {binder_length}")

            
            if not args.skip_mpnn and pdb_stem in binder_seq_map:
                binder_seq = binder_seq_map[pdb_stem]
                print(f"  LigandMPNN top1: {binder_seq[:40]}{'...' if len(binder_seq) > 40 else ''}")
            else:
                binder_seq = binder_seq_from_pdb
                if args.skip_mpnn:
                    print(f" use PDB sequence(--skip_mpnn)")
                else:
                    print(f" notto LigandMPNN sequence(pdb_stem={pdb_stem}),use PDB sequence")

            # Deferred covalent: select attach site, mutate -> G, build per-PDB bonds
            if covalent_cfg is not None:
                rec_chain = receptor_attach_chain(covalent_cfg)
                rec_res = receptor_attach_res_1based(covalent_cfg)
                attach_1based = None
                attach_source = None
                if args.covalent_deferred:
                    attach_1based = detect_attach_site_from_bfactor(
                        str(pdb_path), binder_chain_id
                    )
                    if attach_1based is not None:
                        attach_source = "bfactor"
                    else:
                        attach_1based = lookup_attach_res_from_records(
                            pdb_stem, attach_site_map
                        )
                        if attach_1based is not None:
                            attach_source = "attach_sites.jsonl"
                if attach_1based is None:
                    attach_1based = select_binder_attach_site(
                        pdb_path=str(pdb_path),
                        binder_chain=binder_chain_id,
                        receptor_chain=rec_chain,
                        receptor_res_1based=rec_res,
                        method=covalent_cfg.attach_site_method,
                        ca_min=covalent_cfg.attach_ca_min,
                        ca_max=covalent_cfg.attach_ca_max,
                    )
                    attach_source = "ca_window"
                if attach_1based is None:
                    print(
                        f" ⏭ [covalent] nocrosslinksite(CA-CA notin "
                        f"[{covalent_cfg.attach_ca_min:.1f}, {covalent_cfg.attach_ca_max:.1f}]Å),skip"
                    )
                    continue
                old_aa = binder_seq[attach_1based - 1] if attach_1based <= len(binder_seq) else "?"
                binder_seq = mutate_sequence_to_aa(
                    binder_seq, attach_1based, covalent_cfg.mutate_to
                )
                fixed_binder_indices_0based = [attach_1based - 1]
                yaml_constraints = build_covalent_constraints(
                    covalent_cfg, binder_chain_id, attach_1based
                )
                covalent_midpoint = midpoint_ca(
                    str(pdb_path),
                    binder_chain_id,
                    attach_1based,
                    rec_chain,
                    rec_res,
                )
                print(
                    f"  [covalent] {binder_chain_id}{attach_1based}: "
                    f"{old_aa} -> {covalent_cfg.mutate_to} (via {attach_source}); "
                    f"bonds={len(yaml_constraints)}; B-factor fix index={attach_1based - 1}"
                )
                attach_site_records.append(
                    {
                        "input_stem": pdb_stem,
                        "input_pdb": str(pdb_path),
                        "binder_chain": binder_chain_id,
                        "attach_res_1based": attach_1based,
                        "attach_source": attach_source,
                        "mutate_to": covalent_cfg.mutate_to,
                        "old_aa": old_aa,
                        "receptor_chain": rec_chain,
                        "receptor_res_1based": rec_res,
                        "attach_ca_min": covalent_cfg.attach_ca_min,
                        "attach_ca_max": covalent_cfg.attach_ca_max,
                        "ligand": ligands,
                    }
                )

            
            msa_dir = run_dir / "msa"
            msa_dir.mkdir(parents=True, exist_ok=True)

            binder_msa_csv = None
            receptor_msa_for_feats = receptor_msa_file  

            try:
                print(f" generate fake paired MSA...")
                
                msa_data = prepare_msa_generator_data(
                    pdb_file=str(pdb_path),
                    receptor_chain=receptor_chain_id,
                    binder_chain=binder_chain_id,
                    contact_cutoff=8.0,
                    verbose=(pdb_idx == 1),  
                )

                
                receptor_fake_df, binder_fake_df = generate_dynamic_msa(
                    msa_data=msa_data,
                    binder_seq=binder_seq,
                    num_seqs=args.num_fake_msa,
                    correlation_strength=0.8,
                    save_temp_files=True,
                    output_dir=msa_dir,
                    verbose=(pdb_idx == 1),
                )

                
                fake_receptor_csv = msa_dir / f"receptor_{receptor_chain_id}.csv"
                fake_binder_csv = msa_dir / f"binder_{binder_chain_id}.csv"

                
                if not fake_receptor_csv.exists():
                    receptor_fake_df.to_csv(str(fake_receptor_csv), index=False)
                if not fake_binder_csv.exists():
                    binder_fake_df.to_csv(str(fake_binder_csv), index=False)

                binder_msa_csv = str(fake_binder_csv)
                print(f" Fake MSA generateok: {len(receptor_fake_df)} sequence")

                
                if receptor_msa_file is not None:
                    merged_receptor_csv = msa_dir / "receptor_merged.csv"
                    merge_receptor_msas_to_csv(
                        real_a3m=receptor_msa_file,
                        fake_csv=str(fake_receptor_csv),
                        out_csv=str(merged_receptor_csv),
                    )
                    receptor_msa_for_feats = str(merged_receptor_csv)
                    print(f" Receptor MSA ok: {merged_receptor_csv}")
                else:
                    
                    receptor_msa_for_feats = str(fake_receptor_csv)
                    print(f" no receptor a3m,use fake MSA: {fake_receptor_csv}")

            except Exception as e:
                print(f" Fake MSA generatefailed: {e},notuse fake MSA")
                import traceback
                traceback.print_exc()
                
                binder_msa_csv = None
                receptor_msa_for_feats = receptor_msa_file

            feats, metadata = feature_prep.prepare_complex_features(
                receptor_sequence=receptor_sequence,
                binder_length=binder_length,
                binder_template=binder_seq,  
                receptor_msa_file=receptor_msa_for_feats,  
                binder_msa_file=binder_msa_csv,  
                template_file=str(template_cif),  
                template_chain_id=binder_chain_id,
                template_id=template_id_for_cif,
                template_force=True,
                template_threshold=args.template_threshold,
                ligands=ligands if ligands else None,
                yaml_constraints=yaml_constraints if yaml_constraints else None,
                receptor_chain_id=receptor_chain_id,
                binder_chain_id=binder_chain_id,
            )
            
            
            metadata['receptor_chain_id'] = receptor_chain_id
            metadata['binder_chain_id'] = binder_chain_id
            print(f" Features ok")
        except Exception as e:
            print(f" Features failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        
        print(f" Pairformer get s and z...")
        with torch.no_grad():
            
            s, z = pairformer.forward(feats)
        print(f" s: {s.shape}, z: {z.shape}")

        
        feats_for_diffusion = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                feats_for_diffusion[k] = v.to(args.device)
            else:
                feats_for_diffusion[k] = v

        
        if pdb_idx == 1:
            cpu_keys = []
            non_tensor_keys = []
            for k, v in feats_for_diffusion.items():
                if isinstance(v, torch.Tensor):
                    if str(v.device) == 'cpu':
                        cpu_keys.append(f"{k}: shape={v.shape}")
                else:
                    non_tensor_keys.append(f"{k}: type={type(v).__name__}")
            if cpu_keys:
                print(f" [DEBUG] CPU tensors in feats_for_diffusion: {cpu_keys}")
            else:
                print(f" [DEBUG] All tensors in feats_for_diffusion are on {args.device}")
            if non_tensor_keys:
                print(f"  [DEBUG] Non-tensor keys: {non_tensor_keys}")

        
        print(f" sampling {args.num_samples}...")

        
        
        pdb_start_coords = None
        if args.warm_start:
            try:
                pdb_start_coords = build_start_coords_from_pdb(
                    pdb_path=str(pdb_path),
                    feats=feats_for_diffusion,
                    receptor_chain_id=receptor_chain_id,
                    binder_chain_id=binder_chain_id,
                    device=args.device,
                )
                if covalent_cfg is not None and covalent_midpoint is not None:
                    pdb_start_coords = seed_ligand_start_coords(
                        pdb_start_coords,
                        feats_for_diffusion,
                        covalent_midpoint,
                        protein_asym_count=2,
                    )
            except Exception as e:
                print(f" coordinatesfailed: {e},tonoise")
                import traceback
                traceback.print_exc()
                pdb_start_coords = None

        for sample_idx in range(args.num_samples):
            try:
                use_warm = args.warm_start and pdb_start_coords is not None

                if use_warm:
                    print(f" [] {sample_idx + 1}: "
                          f"sigma={args.warm_start_sigma:.1f} (diffusion)")

                with torch.no_grad():
                    coords = diffusion.forward(
                        s=s.to(args.device),
                        z=z.to(args.device),
                        feats=feats_for_diffusion,
                        num_steps=args.sampling_steps,
                        use_fixed_noise=False,  
                        start_coords=pdb_start_coords if use_warm else None,
                        start_sigma=args.warm_start_sigma if use_warm else None,
                        use_physical_guidance=False,
                    )

                coords_cpu = coords.cpu()

                
                if 'token_to_center_atom' in feats:
                    token_to_center_atom = feats['token_to_center_atom'].cpu()
                    if token_to_center_atom.dim() == 2:
                        token_to_center_atom = token_to_center_atom.unsqueeze(0)
                    all_ca_coords = torch.bmm(token_to_center_atom.float(), coords_cpu)[0]
                elif 'token_to_rep_atom' in feats:
                    token_to_rep_atom = feats['token_to_rep_atom'].cpu()
                    if token_to_rep_atom.dim() == 2:
                        token_to_rep_atom = token_to_rep_atom.unsqueeze(0)
                    all_ca_coords = torch.bmm(token_to_rep_atom.float(), coords_cpu)[0]
                else:
                    raise ValueError("feats must contain 'token_to_center_atom' or 'token_to_rep_atom'")

                
                pdb_file = run_dir / f"structure_round01_sample{sample_idx + 1:02d}.pdb"

                write_coords_to_pdb(
                    coords=coords_cpu,
                    feats=feats,
                    metadata=metadata,
                    output_file=str(pdb_file),
                    ca_coords=all_ca_coords,
                    use_backbone=False,
                    use_full_sidechain=True,
                    cdr_regions=pdb_cdr_regions,
                    fixed_binder_indices=fixed_binder_indices_0based,
                )

                print(f" {sample_idx + 1}/{args.num_samples}: {pdb_file.name}")

            except Exception as e:
                print(f" {sample_idx + 1} generatefailed: {e}")
                import traceback
                traceback.print_exc()

        print(f" {pdb_path.name} -> {run_dir}")

    # ======================================================================
    
    
    
    # ======================================================================
    import shutil
    from dream_boltz2.io.names import flat_relaxed_name, write_name_map

    all_relaxed_dir = out_root / "all_relaxed"
    all_relaxed_dir.mkdir(parents=True, exist_ok=True)
    copy_count = 0
    name_tag = "c" if covalent_cfg is not None else "x"
    used_flat: set = set()
    name_map_records: List[Dict[str, Any]] = []
    for sub_dir in sorted(out_root.iterdir()):
        if not sub_dir.is_dir() or sub_dir.name in (
            "mpnn_sequences", "mpnn_filtered", "all_relaxed"
        ):
            continue
        for pdb_f in sorted(sub_dir.glob("structure_round01_sample*.pdb")):
            
            sample_part = pdb_f.stem.replace("structure_round01_", "")  # "sample03"
            flat_name = flat_relaxed_name(
                sub_dir.name, sample_part, tag=name_tag, used=used_flat
            )
            shutil.copy2(str(pdb_f), str(all_relaxed_dir / flat_name))
            name_map_records.append(
                {
                    "short_name": flat_name,
                    "input_stem": sub_dir.name,
                    "run_pdb": pdb_f.name,
                    "tag": name_tag,
                }
            )
            copy_count += 1

    if name_map_records:
        write_name_map(name_map_records, all_relaxed_dir / "name_map.jsonl")
        write_name_map(name_map_records, out_root / "name_map.jsonl")
        print(f" name_map.jsonl -> {all_relaxed_dir / 'name_map.jsonl'}")

    if covalent_cfg is not None and attach_site_records:
        from dream_boltz2.io.names import compact_design_id as _compact

        for rec in attach_site_records:
            rec["design_id"] = _compact(rec["input_stem"])
            for nm in name_map_records:
                if nm["input_stem"] == rec["input_stem"]:
                    rec["short_name_example"] = nm["short_name"]
                    break
            rec["note"] = (
                "Attach site is Gly + B-factor=100 in relaxed PDBs; "
                "batch_design_sequences detects via B-factor"
            )
        write_attach_sites_jsonl(attach_site_records, out_root / "attach_sites.jsonl")
        write_attach_sites_jsonl(
            attach_site_records, all_relaxed_dir / "attach_sites.jsonl"
        )
        print(f" attach_sites.jsonl -> {out_root / 'attach_sites.jsonl'}")

    print(f"\n allbackbone！output directory: {out_root}")
    print(f" output: {all_relaxed_dir}({copy_count} PDB, tag={name_tag})")
    print(f" for backbone_screen.py:")
    print(f"   python scripts/backbone_screen.py {all_relaxed_dir} --hotspots ... --target-chain ...")


if __name__ == "__main__":
    main()
