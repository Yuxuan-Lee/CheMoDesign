#!/usr/bin/env python3
"""Complete-linkage clustering of binder chains at TM-score 0.6."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _stem(name: str) -> str:
    n = Path(name).stem
    if n.endswith("_model"):
        n = n[: -len("_model")]
    if len(n) > 2 and n[-2] == "_" and n[-1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        n = n[:-2]
    return n


def collect_structure_files(input_dir: Path) -> List[Path]:
    files = []
    for pat in ("*.cif", "*.mmcif", "*.pdb"):
        files.extend(input_dir.glob(pat))
    return sorted(files)


def extract_chain_pdbs(
    input_dir: Path,
    out_dir: Path,
    chain_ids: Sequence[str],
) -> Path:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.PDBIO import PDBIO, Select

    chain_set = set(chain_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    class ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.id in chain_set

    files = collect_structure_files(input_dir)
    if not files:
        raise SystemExit(f"no PDB/CIF:{input_dir}")
    ok = 0
    for f in files:
        suffix = f.suffix.lower()
        parser = PDBParser(QUIET=True) if suffix == ".pdb" else MMCIFParser(QUIET=True)
        structure = parser.get_structure(f.stem, str(f))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(out_dir / f"{f.stem}.pdb"), select=ChainSelect())
        ok += 1
    if ok == 0:
        raise SystemExit('chainextractfailed')
    print(f"[INFO] extractchain {list(chain_ids)}:{ok} -> {out_dir}")
    return out_dir


def run_foldseek_allvsall(
    pdb_dir: Path,
    aln_path: Path,
    tmp_dir: Path,
    foldseek: str,
    threads: int,
) -> None:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    aln_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(list(pdb_dir.glob("*.pdb")))
    max_seqs = max(n + 8, 32)
    cmd = [
        foldseek,
        "easy-search",
        str(pdb_dir),
        str(pdb_dir),
        str(aln_path),
        str(tmp_dir),
        "--alignment-type",
        "2",
        "--exhaustive-search",
        "1",
        "-e",
        "inf",
        "--max-seqs",
        str(max_seqs),
        "--format-output",
        "query,target,alntmscore,qtmscore,ttmscore",
        "--threads",
        str(threads),
    ]
    print("[FoldSeek]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_tm_matrix(aln_path: Path) -> Tuple[List[str], np.ndarray, int]:
    """Return names, TM matrix (min of both directions), and missing-pair count."""
    pairs: Dict[Tuple[str, str], float] = {}
    names_set = set()
    with aln_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            q, t, atm, qtm, ttm, *_ = line.split()
            q, t = _stem(q), _stem(t)
            names_set.add(q)
            names_set.add(t)
            qtm_f = float(qtm)
            ttm_f = float(ttm)
            v = min(qtm_f, ttm_f)
            v = float(np.clip(v, 0.0, 1.0))
            key = (q, t) if q <= t else (t, q)
            pairs[key] = max(pairs.get(key, 0.0), v)
            if q == t:
                pairs[key] = 1.0
    names = sorted(names_set)
    n = len(names)
    idx = {s: i for i, s in enumerate(names)}
    tm = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(tm, 1.0)
    missing = 0
    expected = n * (n - 1) // 2
    have = 0
    for i, a in enumerate(names):
        for j in range(i + 1, n):
            b = names[j]
            key = (a, b) if a <= b else (b, a)
            if key in pairs:
                tm[i, j] = tm[j, i] = pairs[key]
                have += 1
            else:
                missing += 1
    if have + missing != expected:
        raise RuntimeError("pair count mismatch")
    return names, tm, missing


def complete_linkage_clusters(tm: np.ndarray, threshold: float) -> np.ndarray:
    n = tm.shape[0]
    if n == 1:
        return np.array([1], dtype=int)
    dist = np.clip(1.0 - tm, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="complete")
    return fcluster(Z, t=1.0 - threshold, criterion="distance")


def write_outputs(
    out_dir: Path,
    names: List[str],
    labels: np.ndarray,
    tm: np.ndarray,
    threshold: float,
    missing: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = Counter(int(x) for x in labels)
    size_list = sorted(sizes.values(), reverse=True)
    tsv = out_dir / "complete_clusters.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cluster_id", "member", "cluster_size"])
        for name, lab in sorted(zip(names, labels), key=lambda x: (int(x[1]), x[0])):
            w.writerow([int(lab), name, sizes[int(lab)]])
    summary = {
        "n": len(names),
        "n_clusters": len(sizes),
        "sizes": size_list,
        "tm_threshold": threshold,
        "method": "complete-linkage",
        "pairwise_tm": "min(qtmscore, ttmscore)",
        "missing_pairs_treated_as_0": missing,
        "expected_pairs": len(names) * (len(names) - 1) // 2,
    }
    (out_dir / "complete_clusters_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"[OK] n={summary['n']} clusters={summary['n_clusters']} "
        f"sizes={'+'.join(map(str, size_list))} missing_pairs={missing}"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=' TM ()')
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--foldseek-path", required=True)
    p.add_argument("--chain-ids", default="A", help='thischain,,default A')
    p.add_argument("--tm-threshold", type=float, default=0.6)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--reuse-aln",
        action="store_true",
        help=' output_dir/aln.tsv alreadyexistsskip FoldSeek',
    )
    args = p.parse_args(argv)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chain_ids = [c.strip() for c in args.chain_ids.split(",") if c.strip()]

    chain_dir = output_dir / "chain_pdbs"
    extract_chain_pdbs(input_dir, chain_dir, chain_ids)

    aln_path = output_dir / "aln.tsv"
    tmp_dir = output_dir / "foldseek_tmp"
    if not (args.reuse_aln and aln_path.exists()):
        run_foldseek_allvsall(
            chain_dir, aln_path, tmp_dir, args.foldseek_path, args.threads
        )
    names, tm, missing = load_tm_matrix(aln_path)
    labels = complete_linkage_clusters(tm, args.tm_threshold)
    write_outputs(output_dir, names, labels, tm, args.tm_threshold, missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
