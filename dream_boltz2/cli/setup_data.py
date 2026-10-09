"""Download the Boltz-2 checkpoint, molecule library, and example ligands."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from dream_boltz2.assets import DATA_REPO, bundled_boltz_dir, project_root


def _download(repo: str, filename: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest),
        )
    )


def _overlay_ligands(src: Path, mol_dir: Path) -> int:
    if not src.is_dir():
        return 0
    mol_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.glob("*.pkl")):
        shutil.copy2(path, mol_dir / path.name)
        n += 1
    return n


def setup(repo: str = DATA_REPO) -> Path:
    dest = bundled_boltz_dir()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {dest}")
    print(f"Downloading from https://huggingface.co/datasets/{repo}")

    ckpt = dest / "boltz2_conf.ckpt"
    if ckpt.is_file() and ckpt.stat().st_size > 1_000_000_000:
        print(f"Checkpoint already present: {ckpt}")
    else:
        ckpt = _download(repo, "boltz2_conf.ckpt", dest)
        print(f"Checkpoint: {ckpt}")

    archive = dest / "mols.tar"
    mol_dir = dest / "mols"
    if (mol_dir / "ALA.pkl").is_file():
        print(f"Molecule library already present: {mol_dir}")
    else:
        if not archive.is_file():
            archive = _download(repo, "mols.tar", dest)
        print(f"Extracting {archive}")
        with tarfile.open(archive, "r") as tar:
            tar.extractall(dest, filter="data")

    custom = dest / "custom_ligands"
    if not any(custom.glob("*.pkl")):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            allow_patterns="custom_ligands/*.pkl",
            local_dir=str(dest),
        )
    n_pack = _overlay_ligands(custom, mol_dir)
    n_examples = _overlay_ligands(project_root() / "examples" / "ligands", mol_dir)
    print(f"Custom ligands copied into mols/: {max(n_pack, n_examples)}")
    if not (mol_dir / "ALA.pkl").is_file():
        raise FileNotFoundError(f"ALA.pkl missing after extract: {mol_dir}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")
    print("Boltz-2 data pack is ready.")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the CheMoDesign Boltz-2 data pack.")
    parser.add_argument("--repo", default=DATA_REPO, help="Hugging Face dataset id.")
    args = parser.parse_args()
    setup(args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
