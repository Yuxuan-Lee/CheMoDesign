#!/usr/bin/env python3
"""
CIF -> Relax-ready PDB converter for DREAM-boltz2.

AF3/BindCraft output CIF files have arbitrary chain IDs that don't match
DREAM-boltz2 YAML configs.  This script:
  1. Reads the project YAML to get target chain IDs and receptor sequence
  2. Identifies CIF chains by sequence matching against the YAML receptor
  3. Remaps chain IDs to match the YAML config
  4. Applies B-factor annotation (receptor=1.0, binder CDR=0.0, framework=100.0)

Limitation: single receptor + single binder only; homo-oligomers not supported.

Usage:
    python scripts/convert_and_swap.py \
        --yaml_config examples/pd_l1.yaml \
        --input_dir /path/to/cif_files/ \
        --output_dir /path/to/output_pdbs/ \
        --skip_cdr_fix           # optional: skip CDR B-factor annotation
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from Bio.PDB import MMCIFParser, PDBIO

# ---------------------------------------------------------------------------
# Reuse helpers from relax_backbones (YAML parsing) and cdr_utils
# ---------------------------------------------------------------------------
from dream_boltz2.cli.wake import (
    _extract_binder_chain_ids,
    _extract_receptor_chain_ids,
    _extract_ligands_from_yaml,
    _get_receptor_sequence_from_yaml,
)
from dream_boltz2.sequence.cdr import _AA_3TO1, validate_and_fix_cdr_bfactor


# ============================================================================
# YAML loading
# ============================================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML top-level must be dict: {path}")
    return data


# ============================================================================
# CIF sequence extraction
# ============================================================================

def extract_sequences_from_cif(cif_path: str) -> Dict[str, str]:
    """
    Extract one-letter amino acid sequences for every chain in a CIF file.

    Returns:
        {chain_id: sequence} dict.  Non-standard / HETATM residues are skipped.
    """
    parser = MMCIFParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure("cif", cif_path)

    sequences: Dict[str, str] = {}
    for chain in structure[0]:
        seq_chars: List[str] = []
        for residue in chain:
            # Skip water and hetero residues (HETATM)
            hetfield = residue.id[0]
            if hetfield.strip():          # non-blank hetfield -> HETATM
                continue
            resname = residue.get_resname().strip()
            aa = _AA_3TO1.get(resname)
            if aa is not None:
                seq_chars.append(aa)
        if seq_chars:
            sequences[chain.id] = "".join(seq_chars)
    return sequences


# ============================================================================
# Sequence identity
# ============================================================================

def sequence_identity(seq1: str, seq2: str) -> float:
    """Fraction of identical residues.  Handle length diff via max(len)."""
    if not seq1 or not seq2:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / max(len(seq1), len(seq2))


# ============================================================================
# Chain identification by sequence matching
# ============================================================================

def identify_chains(
    cif_sequences: Dict[str, str],
    receptor_seq: str,
    receptor_chain_ids: List[str],
    binder_chain_ids: List[str],
    ligand_defs: List[Dict[str, Any]],
    identity_threshold: float = 0.85,
    min_protein_length: int = 50,
) -> Optional[Dict[str, Tuple[str, str]]]:
    """
    Identify which CIF chain is the receptor and which is the binder
    by comparing sequences against the YAML receptor sequence.

    Returns:
        {cif_chain_id: (role, target_chain_id)} mapping, e.g.:
            {"B": ("receptor", "A"), "A": ("binder", "B")}
        None if no receptor match found.
    """
    target_receptor_id = receptor_chain_ids[0]  # e.g. "A"
    target_binder_id = binder_chain_ids[0]      # e.g. "B"

    # Score each CIF chain against the receptor sequence
    best_receptor_cif_id: Optional[str] = None
    best_identity: float = 0.0

    for cif_id, cif_seq in cif_sequences.items():
        ident = sequence_identity(cif_seq, receptor_seq)
        if ident > best_identity:
            best_identity = ident
            best_receptor_cif_id = cif_id

    if best_identity < identity_threshold or best_receptor_cif_id is None:
        return None

    # Remaining protein chains (length >= min_protein_length) -> binder candidates
    binder_candidates = [
        cid for cid, seq in cif_sequences.items()
        if cid != best_receptor_cif_id and len(seq) >= min_protein_length
    ]

    if len(binder_candidates) == 0:
        print(f"  WARNING: no binder candidate found (all remaining chains too short)")
        return None

    if len(binder_candidates) > 1:
        print(f"  WARNING: multiple binder candidates {binder_candidates}, using first one")

    binder_cif_id = binder_candidates[0]

    mapping: Dict[str, Tuple[str, str]] = {
        best_receptor_cif_id: ("receptor", target_receptor_id),
        binder_cif_id: ("binder", target_binder_id),
    }

    # Remaining chains -> ligand / passthrough
    for cif_id in cif_sequences:
        if cif_id not in mapping:
            mapping[cif_id] = ("ligand", cif_id)

    return mapping


# ============================================================================
# CIF -> PDB conversion with chain remapping & B-factor
# ============================================================================

def convert_cif_to_pdb(
    cif_path: str,
    chain_mapping: Dict[str, Tuple[str, str]],
    output_path: str,
) -> None:
    """
    Parse CIF, remap chain IDs, set B-factors, write PDB.

    B-factor convention:
        receptor  -> 1.0
        binder    -> 0.0  (CDR fix applied later if nanobody)
        ligand    -> 0.0
    """
    parser = MMCIFParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure("cif", cif_path)

    model = structure[0]

    # --- Step 1: Remove chains not in mapping ---
    chain_ids_in_model = [c.id for c in model]
    for cid in chain_ids_in_model:
        if cid not in chain_mapping:
            model.detach_child(cid)

    # --- Step 2: Set B-factors before remapping ---
    for chain in model:
        role, _target_id = chain_mapping[chain.id]
        bfactor = 1.0 if role == "receptor" else 0.0
        for residue in chain:
            for atom in residue:
                atom.set_bfactor(bfactor)

    # --- Step 3: Remap chain IDs (use temp IDs to avoid collision) ---
    # First pass: all to temp IDs
    temp_map: Dict[str, str] = {}  # temp_id -> target_id
    for i, cid in enumerate(list(chain_mapping.keys())):
        _role, target_id = chain_mapping[cid]
        temp_id = f"_{i}"
        if cid in model:
            model[cid].id = temp_id
            temp_map[temp_id] = target_id

    # Second pass: temp IDs to final target IDs
    for temp_id, target_id in temp_map.items():
        if temp_id in model:
            model[temp_id].id = target_id

    # --- Step 4: Write PDB ---
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIF -> Relax-ready PDB converter (YAML-driven chain identification)"
    )
    parser.add_argument(
        "--yaml_config", type=str, required=True,
        help="Project YAML (for example examples/pd_l1.yaml)",
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing .cif files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for output .pdb files",
    )
    parser.add_argument(
        "--skip_cdr_fix", action="store_true",
        help="Skip CDR B-factor annotation (default: apply for nanobody design_type)",
    )
    parser.add_argument(
        "--identity_threshold", type=float, default=0.85,
        help="Min sequence identity to match receptor chain (default: 0.85)",
    )
    args = parser.parse_args()

    # --- Parse YAML ---
    yaml_path = Path(args.yaml_config)
    if not yaml_path.exists():
        print(f"ERROR: YAML config not found: {yaml_path}")
        sys.exit(1)
    project_yaml = _load_yaml(yaml_path)

    receptor_chain_ids = _extract_receptor_chain_ids(project_yaml)
    binder_chain_ids = _extract_binder_chain_ids(project_yaml)
    ligand_defs = _extract_ligands_from_yaml(project_yaml)
    receptor_seq = _get_receptor_sequence_from_yaml(project_yaml, receptor_chain_ids)

    if receptor_seq is None:
        print("ERROR: Could not extract receptor sequence from YAML")
        sys.exit(1)

    design_type = project_yaml.get("dream", {}).get("design_type", "binder")
    is_nanobody = design_type == "nanobody"

    print(f"=== CIF -> PDB Converter ===")
    print(f"  YAML config  : {yaml_path}")
    print(f"  Design type  : {design_type}")
    print(f"  Receptor chain (target): {receptor_chain_ids[0]}")
    print(f"  Binder chain  (target): {binder_chain_ids[0]}")
    print(f"  Receptor seq len: {len(receptor_seq)}")
    print(f"  CDR fix      : {'skip' if args.skip_cdr_fix else ('auto (nanobody)' if is_nanobody else 'skip (not nanobody)')}")
    print()

    # --- Prepare output dir ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Batch processing ---
    cif_files = sorted(glob.glob(os.path.join(args.input_dir, "*.cif")))
    if not cif_files:
        print(f"No .cif files found in {args.input_dir}")
        sys.exit(0)

    success_count = 0
    skip_count = 0
    cdr_fix_count = 0

    for cif_path in cif_files:
        basename = os.path.basename(cif_path)
        output_name = os.path.splitext(basename)[0] + ".pdb"
        output_path = os.path.join(args.output_dir, output_name)

        print(f"[{basename}]")

        # 1. Extract sequences from CIF
        try:
            cif_sequences = extract_sequences_from_cif(cif_path)
        except Exception as e:
            print(f"  ERROR: CIF parse failed: {e}")
            skip_count += 1
            print()
            continue

        if not cif_sequences:
            print(f"  ERROR: No protein chains found in CIF")
            skip_count += 1
            print()
            continue

        print(f"  CIF chains: {', '.join(f'{cid}(len={len(seq)})' for cid, seq in cif_sequences.items())}")

        # 2. Identify chains by sequence matching
        chain_mapping = identify_chains(
            cif_sequences,
            receptor_seq,
            receptor_chain_ids,
            binder_chain_ids,
            ligand_defs,
            identity_threshold=args.identity_threshold,
        )

        if chain_mapping is None:
            best_ident = max(
                (sequence_identity(seq, receptor_seq) for seq in cif_sequences.values()),
                default=0.0,
            )
            print(f"  SKIP: No receptor match (best identity={best_ident:.2f}, threshold={args.identity_threshold})")
            skip_count += 1
            print()
            continue

        # Print mapping
        for cif_id, (role, target_id) in chain_mapping.items():
            ident_str = ""
            if role == "receptor":
                ident_str = f" (identity={sequence_identity(cif_sequences[cif_id], receptor_seq):.2f})"
            print(f"  {cif_id} -> {target_id} ({role}){ident_str}")

        # 3. Convert CIF -> PDB with remapped chains
        try:
            convert_cif_to_pdb(cif_path, chain_mapping, output_path)
        except Exception as e:
            print(f"  ERROR: Conversion failed: {e}")
            skip_count += 1
            print()
            continue

        # 4. CDR B-factor annotation (nanobody only)
        if is_nanobody and not args.skip_cdr_fix:
            try:
                is_ok, cdr_regions, template = validate_and_fix_cdr_bfactor(
                    output_path, binder_chain=binder_chain_ids[0],
                )
                if template is not None:
                    print(f"  CDR fix: template={template}, regions={cdr_regions}, {'consistent' if is_ok else 'repaired'}")
                    cdr_fix_count += 1
                else:
                    print(f"  CDR fix: no template matched, binder B-factor kept at 0.0")
            except Exception as e:
                print(f"  WARNING: CDR fix failed ({e}), binder B-factor kept at 0.0")

        print(f"  OK -> {output_path}")
        success_count += 1
        print()

    # --- Summary ---
    print(f"=== Done ===")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    if is_nanobody and not args.skip_cdr_fix:
        print(f"  CDR annotated: {cdr_fix_count}")
    print(f"  Output dir: {args.output_dir}")
    print()
    print(f"B-factor convention:")
    print(f"  Receptor      = 1.0  (fixed)")
    if is_nanobody and not args.skip_cdr_fix:
        print(f"  Binder framework = 100.0 (fixed by MPNN)")
        print(f"  Binder CDR    = 0.0  (designable)")
    else:
        print(f"  Binder        = 0.0  (designable)")


if __name__ == "__main__":
    main()
