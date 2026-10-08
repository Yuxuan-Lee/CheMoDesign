"""Concatenate a real receptor MSA with synthetic binder rows."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(str(path), "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_a3m_sequences(a3m_path: str | Path, *, max_seqs: Optional[int] = None) -> Iterator[str]:
    """iter a3m sequences."""
    a3m_path = Path(a3m_path)
    n = 0
    with _open_text(a3m_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(">"):
                continue
            yield line
            n += 1
            if max_seqs is not None and n >= max_seqs:
                break


def a3m_to_csv_df(
    a3m_path: str | Path,
    *,
    max_seqs: Optional[int] = None,
) -> pd.DataFrame:
    """a3m to csv df."""
    seqs = list(iter_a3m_sequences(a3m_path, max_seqs=max_seqs))
    return pd.DataFrame({"sequence": seqs, "key": [""] * len(seqs)})


def load_boltz_csv_msa(path: str | Path) -> pd.DataFrame:
    """load boltz csv msa."""
    df = pd.read_csv(path)
    cols = set(df.columns)
    if cols != {"sequence", "key"}:
        raise ValueError(f"Invalid CSV format: {path}, expected columns {{'sequence','key'}}, got {sorted(cols)}")
    
    return df[["sequence", "key"]]


def merge_receptor_msas_to_csv(
    *,
    real_a3m: str | Path,
    fake_csv: str | Path,
    out_csv: str | Path,
    max_real: Optional[int] = None,
    max_fake: Optional[int] = None,
    dedup_by_sequence: bool = True,
) -> Path:
    """merge receptor msas to csv."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    real_df = a3m_to_csv_df(real_a3m, max_seqs=max_real)
    fake_df = load_boltz_csv_msa(fake_csv)
    if max_fake is not None:
        fake_df = fake_df.head(max_fake).copy()

    
    if real_df["key"].astype(str).str.strip().ne("").any():
        raise ValueError('real a3m convert key notcontains taxonomy info(asempty)')

    
    
    if fake_df["key"].isna().any() or fake_df["key"].astype(str).str.strip().eq("").any():
        raise ValueError(
            f"fake_csv inexistsempty key(taxonomy_id).this paired MSA vsinfo:{fake_csv}"
        )

    merged = pd.concat([real_df, fake_df], ignore_index=True)

    if dedup_by_sequence and not merged.empty:
        
        key = merged["sequence"].astype(str).str.replace("-", "", regex=False).str.upper()
        merged = merged.loc[~key.duplicated(keep="first")].copy()

    merged.to_csv(out_csv, index=False)
    return out_csv


