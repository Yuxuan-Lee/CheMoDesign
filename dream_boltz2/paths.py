"""Import paths for running the command-line tools without an editable install."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_import_paths() -> Path:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    boltz_src = os.environ.get("BOLTZ_SRC")
    if boltz_src and boltz_src not in sys.path:
        sys.path.insert(0, boltz_src)
    return root
