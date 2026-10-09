"""Upload the Boltz-2 checkpoint, mols.tar, and example ligands to Hugging Face.

Requires a write token from https://huggingface.co/settings/tokens
(huggingface-cli login, or HF_TOKEN). Does not print the token.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dream_boltz2.assets import DATA_REPO, project_root


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(p) for p in paths))


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the CheMoDesign data pack.")
    parser.add_argument("--repo", default=DATA_REPO)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if not token_file.is_file():
            raise SystemExit("Log in first: huggingface-cli login")

    home = Path.home() / ".boltz"
    ckpt = _first_existing([
        home / "boltz2_conf.ckpt",
        home / "checkpoints" / "boltz2_conf.ckpt",
    ])
    archive = _first_existing([
        home / "mols.tar",
        home / "checkpoints" / "mols.tar",
    ])
    ligands = project_root() / "examples" / "ligands"
    license_boltz = project_root() / "third_party" / "LICENSES" / "LICENSE.boltz"
    card = project_root() / "pack" / "README.md"

    stage = Path(tempfile.mkdtemp(prefix="chemo_pack_"))
    try:
        shutil.copy2(card, stage / "README.md")
        shutil.copy2(license_boltz, stage / "LICENSE.boltz")
        ligand_dest = stage / "custom_ligands"
        ligand_dest.mkdir()
        for path in sorted(ligands.glob("*.pkl")):
            shutil.copy2(path, ligand_dest / path.name)
        print(f"Staging small files in {stage}")
        print(f"Checkpoint: {ckpt}")
        print(f"Molecule archive: {archive}")
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo,
            repo_type="dataset",
            folder_path=str(stage),
            commit_message="Add CheMoDesign data card, license, and custom ligands.",
        )
        api.upload_file(
            repo_id=args.repo,
            repo_type="dataset",
            path_or_fileobj=str(archive),
            path_in_repo="mols.tar",
            commit_message="Add the Boltz-2 molecule library.",
        )
        api.upload_file(
            repo_id=args.repo,
            repo_type="dataset",
            path_or_fileobj=str(ckpt),
            path_in_repo="boltz2_conf.ckpt",
            commit_message="Add the Boltz-2 confidence checkpoint.",
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"Published https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
