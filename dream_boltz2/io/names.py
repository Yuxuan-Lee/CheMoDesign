"""Short, stable PDB basenames for multi-stage DREAM pipelines.

Avoids ENAMETOOLONG from nested `{stem}_relaxed_{stem}_...` prefixes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

_ROUND_SAMPLE = re.compile(r"round0*(\d+).*?sample0*(\d+)", re.IGNORECASE)
_SHORT_RS = re.compile(r"r(\d+)s(\d+)", re.IGNORECASE)
_SAMPLE_NUM = re.compile(r"(\d+)")


def compact_design_id(stem: str) -> str:
    """
    First round/sample identity in stem -> r01s03.
    Falls back to short hash if no pattern found.
    """
    m = _ROUND_SAMPLE.search(stem)
    if m:
        return f"r{int(m.group(1)):02d}s{int(m.group(2)):02d}"
    m2 = _SHORT_RS.search(stem)
    if m2:
        return f"r{int(m2.group(1)):02d}s{int(m2.group(2)):02d}"
    return "h" + hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]


def sample_index(sample_part: str) -> int:
    """'sample03' / '03' / 'x03' -> 3."""
    m = _SAMPLE_NUM.search(sample_part or "")
    return int(m.group(1)) if m else 1


def flat_relaxed_name(
    input_stem: str,
    sample_part: str,
    tag: str = "x",
    used: Optional[Set[str]] = None,
) -> str:
    """
    Flat all_relaxed basename, e.g. r01s03_x02.pdb / r01s03_c01.pdb.

    tag: 'x' = protein relax, 'c' = covalent relax.
    """
    used = used if used is not None else set()
    base = compact_design_id(input_stem)
    idx = sample_index(sample_part)
    name = f"{base}_{tag}{idx:02d}.pdb"
    if name not in used:
        used.add(name)
        return name
    # collision: include short hash of full stem
    h = hashlib.md5(input_stem.encode("utf-8")).hexdigest()[:4]
    name = f"{base}_{tag}{idx:02d}_{h}.pdb"
    n = 2
    while name in used:
        name = f"{base}_{tag}{idx:02d}_{h}{n}.pdb"
        n += 1
    used.add(name)
    return name


def write_name_map(records: Iterable[Dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
