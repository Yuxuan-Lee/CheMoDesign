#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mask the covalent attachment residue before sequence export."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_HASH_SUFFIX = re.compile(r"^(.+_c\d+)_[0-9a-fA-F]{4}$")
_C_SUFFIX = re.compile(r"^(.+)_c\d+")


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = ""
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        if header:
            records.append((header, "".join(seq_parts)))
    return records


def write_fasta(path: Path, records: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")


def load_name_mapping(path: Optional[Path]) -> Dict[str, str]:
    """NewID (taskN_bbM_seqK) -> TaskName (pdb stem)."""
    out: Dict[str, str] = {}
    if path is None or not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            new_id = (row.get("NewID") or "").strip()
            task = (row.get("TaskName") or "").strip()
            if new_id and task:
                out[new_id] = task
    return out


def load_fixed_positions(path: Optional[Path]) -> Dict[str, Dict[str, List[int]]]:
    out: Dict[str, Dict[str, List[int]]] = {}
    if path is None or not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict) or len(rec) != 1:
                continue
            stem, chains = next(iter(rec.items()))
            if isinstance(chains, dict):
                out[str(stem)] = {
                    str(ch): [int(x) for x in vals]
                    for ch, vals in chains.items()
                    if isinstance(vals, (list, tuple))
                }
    return out


def load_attach_index(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    # Prefer project helper when available; fall back to local parse.
    try:
        repo = Path(__file__).resolve().parents[2]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from dream_boltz2.covalent.deferred import load_attach_sites_jsonl

        return load_attach_sites_jsonl(path)
    except Exception:
        out: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for key in ("input_stem", "output_stem", "flat_prefix", "design_id"):
                    if rec.get(key):
                        out[str(rec[key])] = rec
                example = rec.get("short_name_example")
                if example:
                    stem = Path(str(example)).stem
                    out[stem] = rec
        return out


def _candidate_keys(stem: str) -> List[str]:
    keys = [stem]
    m = _HASH_SUFFIX.match(stem)
    if m:
        keys.append(m.group(1))
    m2 = _C_SUFFIX.match(stem)
    if m2:
        keys.append(m2.group(1))
    # unique preserve order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def resolve_attach(
    stem: str,
    attach_index: Dict[str, Dict[str, Any]],
    fixed_positions: Dict[str, Dict[str, List[int]]],
    binder_chain: str,
) -> Tuple[Optional[int], Optional[str], str, Optional[Dict[str, Any]]]:
    """
    Returns (attach_res_1based, expected_aa, source, record_or_none).

    Prefer fixed_positions (per covalent PDB / LigandMPNN stem). attach_sites
    design_id can collide across x01/x02 inputs that share a compact id but
    different attach residues.
    """
    for key in _candidate_keys(stem):
        if key in fixed_positions:
            chains = fixed_positions[key]
            positions = chains.get(binder_chain) or []
            if not positions and len(chains) == 1:
                positions = next(iter(chains.values()))
            if positions:
                return int(positions[0]), "G", f"fixed_positions:{key}", None

    design_id = None
    m2 = _C_SUFFIX.match(stem)
    if m2:
        design_id = m2.group(1)

    # Prefer attach_sites keys that are not the bare design_id
    for key in _candidate_keys(stem):
        if design_id is not None and key == design_id:
            continue
        if key in attach_index:
            rec = attach_index[key]
            pos = int(rec["attach_res_1based"])
            expected = str(rec.get("mutate_to") or "G").upper()
            return pos, expected, f"attach_sites:{key}", rec

    # Last resort: design_id (may be ambiguous across x01/x02)
    if design_id and design_id in attach_index:
        rec = attach_index[design_id]
        pos = int(rec["attach_res_1based"])
        expected = str(rec.get("mutate_to") or "G").upper()
        return pos, expected, f"attach_sites_design_id:{design_id}", rec

    return None, None, "unresolved", None


def resolve_stem_for_header(
    header: str,
    name_map: Dict[str, str],
) -> str:
    hid = header.split()[0]
    if hid in name_map:
        return name_map[hid]
    # raw LigandMPNN / unrenamed headers often start with stem
    if "_processed" in hid:
        return hid.split("_processed")[0]
    return hid


def mask_sequence(
    seq: str,
    pos_1based: int,
    expected_aa: str,
    replace_with: str = "X",
) -> Tuple[str, str, Optional[str]]:
    """
    Returns (new_seq, status, detail).
    status: ok | already_x | mismatch | out_of_range
    """
    if pos_1based < 1 or pos_1based > len(seq):
        return seq, "out_of_range", f"pos={pos_1based} len={len(seq)}"
    i = pos_1based - 1
    aa = seq[i].upper()
    if aa == replace_with.upper():
        return seq, "already_x", None
    if aa != expected_aa.upper():
        return seq, "mismatch", f"found={aa} expected={expected_aa}"
    chars = list(seq)
    chars[i] = replace_with.upper()
    return "".join(chars), "ok", None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Replace recorded covalent attach sites in binder FASTA with X"
    )
    ap.add_argument("--fasta", required=True, type=Path, help="Input FASTA (e.g. best_designs.fa)")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output FASTA (default: overwrite --fasta after writing .bak)",
    )
    ap.add_argument(
        "--attach_sites",
        type=Path,
        default=None,
        help="attach_sites.jsonl from covalent relax",
    )
    ap.add_argument(
        "--fixed_positions",
        type=Path,
        default=None,
        help="fixed_positions.jsonl from batch_design_sequences",
    )
    ap.add_argument(
        "--name_mapping",
        type=Path,
        default=None,
        help="name_mapping.tsv from filter_best_designs --rename --name_map",
    )
    ap.add_argument("--binder_chain", default="B", help="Chain id in fixed_positions (default B)")
    ap.add_argument("--replace_with", default="X", help="Replacement letter (default X)")
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write per-sequence JSONL report (default: <output>.covalent_x_report.jsonl)",
    )
    ap.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not write .bak when overwriting input",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any site unresolved or AA mismatch",
    )
    args = ap.parse_args()

    fasta_in = args.fasta.resolve()
    if not fasta_in.exists():
        print(f" FASTA notexists: {fasta_in}", file=sys.stderr)
        return 2

    out_path = args.output.resolve() if args.output else fasta_in
    report_path = (
        args.report.resolve()
        if args.report
        else out_path.with_suffix(out_path.suffix + ".covalent_x_report.jsonl")
    )

    attach_index = load_attach_index(args.attach_sites)
    fixed_positions = load_fixed_positions(args.fixed_positions)
    name_map = load_name_mapping(args.name_mapping)

    if not attach_index and not fixed_positions:
        print(
            ' need --attach_sites or --fixed_positions',
            file=sys.stderr,
        )
        return 2

    records = parse_fasta(fasta_in)
    out_records: List[Tuple[str, str]] = []
    report_rows: List[Dict[str, Any]] = []
    n_ok = n_already = n_mismatch = n_unresolved = n_oor = 0

    for header, seq in records:
        stem = resolve_stem_for_header(header, name_map)
        pos, expected, source, _rec = resolve_attach(
            stem, attach_index, fixed_positions, args.binder_chain
        )
        row: Dict[str, Any] = {
            "header": header,
            "stem": stem,
            "source": source,
            "attach_res_1based": pos,
            "expected_aa": expected,
            "old_aa": None,
            "status": None,
            "detail": None,
        }
        if pos is None or expected is None:
            n_unresolved += 1
            row["status"] = "unresolved"
            out_records.append((header, seq))
            report_rows.append(row)
            continue

        row["old_aa"] = seq[pos - 1].upper() if 1 <= pos <= len(seq) else None
        new_seq, status, detail = mask_sequence(
            seq, pos, expected, replace_with=args.replace_with
        )
        row["status"] = status
        row["detail"] = detail
        if status == "ok":
            n_ok += 1
        elif status == "already_x":
            n_already += 1
        elif status == "mismatch":
            n_mismatch += 1
        elif status == "out_of_range":
            n_oor += 1
        out_records.append((header, new_seq if status == "ok" else seq))
        report_rows.append(row)

    if out_path == fasta_in and not args.no_backup:
        bak = Path(str(fasta_in) + ".bak_before_covalent_x")
        if not bak.exists():
            shutil.copy2(fasta_in, bak)
            print(f" backup: {bak}")

    write_fasta(out_path, out_records)
    with report_path.open("w", encoding="utf-8") as rf:
        for row in report_rows:
            rf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 72)
    print(f"input: {fasta_in}")
    print(f"output: {out_path}")
    print(f": {report_path}")
    print(f"sequence: {len(records)}")
    print(f" as {args.replace_with}: {n_ok}")
    print(f" already {args.replace_with}: {n_already}")
    print(f" site AA not: {n_mismatch}")
    print(f" site: {n_oor}")
    print(f" notparsetosite: {n_unresolved}")
    print("=" * 72)

    bad = n_mismatch + n_unresolved + n_oor
    if args.strict and bad:
        print(f" strict: {bad} not", file=sys.stderr)
        return 1
    if bad:
        print(f" {bad} not()")
    else:
        print(' (oralready X)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
