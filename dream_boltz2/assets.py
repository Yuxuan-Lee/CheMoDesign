"""Locations of the Boltz-2 checkpoint and molecule library."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DATA_REPO = "Yuxuan-Lee/CheMoDesign-data"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundled_boltz_dir() -> Path:
    return project_root() / "third_party" / "boltz"


def _has_mols(cache: Path) -> bool:
    return (cache / "mols" / "ALA.pkl").is_file() or (
        cache / "checkpoints" / "mols" / "ALA.pkl"
    ).is_file()


def resolve_boltz_cache() -> Path:
    """Directory that holds boltz2_conf.ckpt and mols/.

    Preference is BOLTZ_CACHE, then the data pack under third_party/boltz,
    then the official Boltz cache at ~/.boltz.
    """
    env = os.environ.get("BOLTZ_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    bundled = bundled_boltz_dir()
    home = Path.home() / ".boltz"
    if _has_mols(bundled) or (bundled / "boltz2_conf.ckpt").is_file():
        return bundled
    if (
        _has_mols(home)
        or (home / "boltz2_conf.ckpt").is_file()
        or (home / "checkpoints" / "boltz2_conf.ckpt").is_file()
    ):
        return home
    return bundled


def resolve_mol_dir(cache: Optional[Path] = None) -> Path:
    """Molecule pickle directory. Boltz-2 uses mols/, not ccd.pkl."""
    cache = resolve_boltz_cache() if cache is None else cache
    for candidate in (cache / "mols", cache / "checkpoints" / "mols"):
        if (candidate / "ALA.pkl").is_file():
            return candidate
    return cache / "mols"


def resolve_ccd_pkl(cache: Optional[Path] = None) -> Optional[Path]:
    """Boltz-1 bundled dictionary, if a previous download left one behind."""
    cache = resolve_boltz_cache() if cache is None else cache
    for candidate in (
        cache / "ccd.pkl",
        cache / "checkpoints" / "ccd.pkl",
        Path.home() / ".boltz" / "ccd.pkl",
    ):
        if candidate.is_file():
            return candidate
    return None


def resolve_checkpoint() -> Optional[Path]:
    """Boltz-2 confidence checkpoint, or None if it has not been downloaded."""
    env = os.environ.get("BOLTZ_CHECKPOINT")
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else path
    cache = resolve_boltz_cache()
    for candidate in (
        cache / "boltz2_conf.ckpt",
        cache / "checkpoints" / "boltz2_conf.ckpt",
        bundled_boltz_dir() / "boltz2_conf.ckpt",
        Path.home() / ".boltz" / "boltz2_conf.ckpt",
        Path.home() / ".boltz" / "checkpoints" / "boltz2_conf.ckpt",
    ):
        if candidate.is_file():
            return candidate
    return None
