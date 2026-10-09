"""Locations of the Boltz-2 checkpoint and molecule library."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DATA_REPO = "Yuxuan-Li/CheMoDesign-data"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundled_boltz_dir() -> Path:
    return project_root() / "third_party" / "boltz"


def resolve_boltz_cache() -> Path:
    """Directory of boltz2_conf.ckpt and mols/, relative to this repository."""
    env = os.environ.get("BOLTZ_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return bundled_boltz_dir()


def resolve_mol_dir(cache: Optional[Path] = None) -> Path:
    """Molecule pickle directory. Boltz-2 reads third_party/boltz/mols/."""
    cache = resolve_boltz_cache() if cache is None else cache
    return cache / "mols"


def resolve_ccd_pkl(cache: Optional[Path] = None) -> Optional[Path]:
    """Optional Boltz-1 dictionary sitting next to the checkpoint. Not required."""
    cache = resolve_boltz_cache() if cache is None else cache
    path = cache / "ccd.pkl"
    return path if path.is_file() else None


def resolve_checkpoint() -> Optional[Path]:
    """third_party/boltz/boltz2_conf.ckpt, or None before setup_data."""
    env = os.environ.get("BOLTZ_CHECKPOINT")
    if env:
        return Path(env).expanduser()
    path = resolve_boltz_cache() / "boltz2_conf.ckpt"
    return path if path.is_file() else None
