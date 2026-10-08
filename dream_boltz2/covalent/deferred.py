"""
Deferred covalent mode helpers for DREAM-boltz2.

Design (optimize_s_z): no ligand, no bonds (covalent stays in dream.covalent).

relax_backbones, when YAML has covalent_mode: deferred:
  Round 1: LigandMPNN -> pick binder attach by CA-CA -> inject ligand + bonds,
           mutate -> G, relax (B-factor=100 on the Gly)
  Round 2 (--covalent_deferred): LigandMPNN again (Gly stays fixed via B-factor),
           reuse the attach site, relax again with the same covalent constraints
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dream_boltz2.config import Constraint, parse_constraint


@dataclass
class CovalentConfig:
    """Parsed dream.covalent block (deferred injection)."""

    enabled: bool = False
    mode: str = "off"  # "off" | "deferred"
    ligand_id: str = "C"
    ligand_ccd: str = ""
    # Receptor side is fixed in YAML, e.g. [A, 164, CA] <-> [C, 1, C10]
    receptor_bond_atom1: Tuple[Any, ...] = ()
    receptor_bond_atom2: Tuple[Any, ...] = ()
    # Binder side: residue auto-selected; only ligand atom + binder atom name are fixed
    binder_ligand_atom: Tuple[Any, ...] = ()  # e.g. [C, 1, C5]
    binder_atom_name: str = "CA"
    mutate_to: str = "G"
    
    attach_site_method: str = "ca_window"
    attach_ca_min: float = 8.0
    attach_ca_max: float = 12.0
    protected_ligands: List[str] = field(default_factory=list)

    def ligands(self) -> List[Dict[str, Any]]:
        if not self.ligand_ccd:
            return []
        return [{"id": self.ligand_id, "ccd": self.ligand_ccd}]


def is_covalent_deferred(project_yaml: Dict[str, Any]) -> bool:
    dream = project_yaml.get("dream", {}) or {}
    mode = str(dream.get("covalent_mode", "off")).strip().lower()
    return mode in ("deferred", "late", "post_relax")


def parse_covalent_config(project_yaml: Dict[str, Any]) -> CovalentConfig:
    """Parse dream.covalent (+ covalent_mode). Returns disabled config if off/missing."""
    dream = project_yaml.get("dream", {}) or {}
    mode = str(dream.get("covalent_mode", "off")).strip().lower()
    cov = dream.get("covalent") or {}
    if not isinstance(cov, dict):
        cov = {}

    enabled = mode in ("deferred", "late", "post_relax") and bool(cov)
    cfg = CovalentConfig(enabled=enabled, mode=mode if enabled else "off")
    if not enabled:
        return cfg

    lig = cov.get("ligand") or {}
    cfg.ligand_id = str(lig.get("id", "C"))
    cfg.ligand_ccd = str(lig.get("ccd", "") or "")

    rb = cov.get("receptor_bond") or {}
    atom1 = rb.get("atom1") or []
    atom2 = rb.get("atom2") or []
    cfg.receptor_bond_atom1 = tuple(atom1)
    cfg.receptor_bond_atom2 = tuple(atom2)

    bla = cov.get("binder_ligand_atom") or cov.get("binder_bond_ligand_atom") or []
    cfg.binder_ligand_atom = tuple(bla)
    cfg.binder_atom_name = str(cov.get("binder_atom", cov.get("binder_bond_atom", "CA")))
    cfg.mutate_to = str(cov.get("mutate_to", "G")).upper()[:1] or "G"
    cfg.attach_site_method = str(cov.get("attach_site_method", "ca_window"))
    cfg.attach_ca_min = float(cov.get("attach_ca_min", 8.0))
    cfg.attach_ca_max = float(cov.get("attach_ca_max", 12.0))
    prots = cov.get("protected_ligands") or []
    if isinstance(prots, str):
        prots = [prots]
    cfg.protected_ligands = [str(x) for x in prots]
    if cfg.ligand_ccd and cfg.ligand_ccd not in cfg.protected_ligands:
        cfg.protected_ligands.append(cfg.ligand_ccd)

    return cfg


def _ca_coords_from_pdb(
    pdb_path: str,
    chain_id: str,
) -> Dict[int, Tuple[float, float, float]]:
    """1-based residue number -> CA xyz."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"chain {chain_id} not in {pdb_path}")
    out: Dict[int, Tuple[float, float, float]] = {}
    for res in model[chain_id].get_residues():
        hetflag = res.id[0]
        if hetflag == "W":
            continue
        if "CA" not in res:
            continue
        resseq = int(res.id[1])
        xyz = res["CA"].get_vector().get_array()
        out[resseq] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return out


def detect_attach_site_from_bfactor(
    pdb_path: str,
    binder_chain: str,
    bfactor_min: float = 99.0,
) -> Optional[int]:
    """Return 1-based binder residue if exactly one CA has B-factor >= min.

    First-round covalent relax writes B-factor=100 on the attach Gly.
    Multiple hits (e.g. nanobody framework) -> None so the caller can fall back.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]
    if binder_chain not in model:
        return None
    hits: List[int] = []
    for res in model[binder_chain].get_residues():
        if res.id[0] == "W":
            continue
        if "CA" not in res:
            continue
        if float(res["CA"].bfactor) >= bfactor_min:
            hits.append(int(res.id[1]))
    if len(hits) == 1:
        return hits[0]
    return None


def collect_attach_site_maps(input_dir: Path, max_parents: int = 4) -> Dict[str, Dict[str, Any]]:
    """Merge attach_sites.jsonl walking from input_dir up a few parents."""
    merged: Dict[str, Dict[str, Any]] = {}
    p = Path(input_dir).resolve()
    seen = set()
    for _ in range(max(1, int(max_parents))):
        cand = p / "attach_sites.jsonl"
        key = str(cand)
        if cand.exists() and key not in seen:
            seen.add(key)
            merged.update(load_attach_sites_jsonl(cand))
        if p.parent == p:
            break
        p = p.parent
    return merged


def lookup_attach_res_from_records(
    pdb_stem: str,
    records: Dict[str, Dict[str, Any]],
) -> Optional[int]:
    rec = records.get(pdb_stem)
    if not rec:
        return None
    val = rec.get("attach_res_1based")
    if val is None:
        return None
    return int(val)


def select_binder_attach_site(
    pdb_path: str,
    binder_chain: str,
    receptor_chain: str,
    receptor_res_1based: int,
    method: str = "ca_window",
    ca_min: float = 8.0,
    ca_max: float = 12.0,
) -> Optional[int]:
    """select binder attach site."""
    method = (method or "ca_window").strip().lower()
    binder_ca = _ca_coords_from_pdb(pdb_path, binder_chain)
    if not binder_ca:
        raise RuntimeError(f"no binder CA in {pdb_path} chain {binder_chain}")

    receptor_ca = _ca_coords_from_pdb(pdb_path, receptor_chain)
    if receptor_res_1based not in receptor_ca:
        print(
            f"  [covalent] receptor {receptor_chain}{receptor_res_1based} "
            f"CA missing -> skip attach"
        )
        return None
    rx, ry, rz = receptor_ca[receptor_res_1based]

    # All binder CA distances to receptor attach CA
    dist_by_res: Dict[int, float] = {}
    for res, (x, y, z) in binder_ca.items():
        dist_by_res[res] = ((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2) ** 0.5

    if method in ("ca_window", "fsyh", "placeholder"):
        
        in_window = [
            (res, d) for res, d in dist_by_res.items() if ca_min <= d <= ca_max
        ]
        if not in_window:
            nearest_res, nearest_d = min(dist_by_res.items(), key=lambda kv: kv[1])
            print(
                f"  [covalent/ca_window] no binder CA in [{ca_min:.1f}, {ca_max:.1f}]Å "
                f"of {receptor_chain}{receptor_res_1based}; "
                f"nearest={binder_chain}{nearest_res} ({nearest_d:.2f}Å) -> skip"
            )
            return None
        best_res, best_d = min(in_window, key=lambda kv: kv[1])
        print(
            f"  [covalent/ca_window] binder attach = {binder_chain}{best_res} "
            f"(CA-CA {best_d:.2f}Å ∈ [{ca_min:.1f}, {ca_max:.1f}]Å to "
            f"{receptor_chain}{receptor_res_1based}; "
            f"{len(in_window)} candidates in window)"
        )
        return int(best_res)

    if method in ("closest", "closest_ca"):
        best_res, best_d = min(dist_by_res.items(), key=lambda kv: kv[1])
        print(
            f"  [covalent/closest] binder attach = {binder_chain}{best_res} "
            f"(dist={best_d:.2f}Å to {receptor_chain}{receptor_res_1based})"
        )
        return int(best_res)

    raise NotImplementedError(
        f"attach_site_method={method!r} not implemented; "
        f"use 'ca_window' (FSYH: 8-12Å) or 'closest'"
    )


def mutate_sequence_to_aa(sequence: str, res_1based: int, aa: str = "G") -> str:
    """Mutate 1-based position to aa (default Gly)."""
    if res_1based < 1 or res_1based > len(sequence):
        raise IndexError(
            f"attach site {res_1based} out of range for sequence length {len(sequence)}"
        )
    aa = (aa or "G").upper()[:1]
    chars = list(sequence)
    chars[res_1based - 1] = aa
    return "".join(chars)


def build_covalent_constraints(
    cov: CovalentConfig,
    binder_chain: str,
    binder_res_1based: int,
) -> List[Constraint]:
    """Build bond Constraint list: binder↔ligand + receptor↔ligand."""
    raw: List[Dict[str, Any]] = []

    if len(cov.binder_ligand_atom) == 3:
        raw.append(
            {
                "bond": {
                    "atom1": [binder_chain, binder_res_1based, cov.binder_atom_name],
                    "atom2": list(cov.binder_ligand_atom),
                }
            }
        )

    if len(cov.receptor_bond_atom1) == 3 and len(cov.receptor_bond_atom2) == 3:
        raw.append(
            {
                "bond": {
                    "atom1": list(cov.receptor_bond_atom1),
                    "atom2": list(cov.receptor_bond_atom2),
                }
            }
        )

    return [parse_constraint(item) for item in raw]


def receptor_attach_res_1based(cov: CovalentConfig) -> int:
    if len(cov.receptor_bond_atom1) < 2:
        raise ValueError("covalent.receptor_bond.atom1 must be [chain, res, atom]")
    return int(cov.receptor_bond_atom1[1])


def receptor_attach_chain(cov: CovalentConfig) -> str:
    if len(cov.receptor_bond_atom1) < 1:
        raise ValueError("covalent.receptor_bond.atom1 must be [chain, res, atom]")
    return str(cov.receptor_bond_atom1[0])


def write_attach_sites_jsonl(
    records: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_attach_sites_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    """Map pdb stem / design_id / short flat name -> record."""
    path = Path(path)
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in (
                "input_stem",
                "output_stem",
                "flat_prefix",
                "design_id",
                "short_name",
            ):
                if key in rec and rec[key]:
                    out[str(rec[key])] = rec
            example = rec.get("short_name_example")
            if example:
                stem = Path(str(example)).stem
                out[stem] = rec
                # also index without optional screen-hash suffix: foo_c01_a1b2 -> foo_c01
                if len(stem) > 5 and stem[-5] == "_" and all(
                    c in "0123456789abcdef" for c in stem[-4:]
                ):
                    out[stem[:-5]] = rec
    return out


def seed_ligand_start_coords(
    start_coords,  # torch.Tensor [1, N_atoms, 3]
    feats: Dict[str, Any],
    center_xyz: Tuple[float, float, float],
    protein_asym_count: int = 2,
    spread: float = 1.5,
):
    """
    Place ligand atoms (asym indices beyond protein chains) near center_xyz.
    Used when input PDB has no ligand but feats include a ligand entity.
    """
    import torch

    asym_id = feats.get("asym_id", None)
    if asym_id is None:
        return start_coords
    if asym_id.dim() >= 2:
        asym_id = asym_id[0]
    asym_id = asym_id.cpu()

    atom_to_token = feats["atom_to_token"].cpu()
    if atom_to_token.dim() == 3:
        atom_to_token = atom_to_token[0]
    atom_to_token_idx = atom_to_token.argmax(dim=-1)

    atom_mask = feats["atom_pad_mask"].cpu()
    if atom_mask.dim() == 2:
        atom_mask = atom_mask[0]

    unique_asym = asym_id.unique(sorted=True)
    if len(unique_asym) <= protein_asym_count:
        return start_coords

    ligand_asyms = unique_asym[protein_asym_count:]
    device = start_coords.device
    center = torch.tensor(center_xyz, dtype=start_coords.dtype, device=device)

    for asym_val in ligand_asyms:
        tok_mask = asym_id == asym_val
        tok_indices = torch.where(tok_mask)[0]
        if len(tok_indices) == 0:
            continue
        atom_is_lig = tok_mask[atom_to_token_idx] & (atom_mask > 0.5)
        n = int(atom_is_lig.sum().item())
        if n == 0:
            continue
        noise = spread * torch.randn(n, 3, dtype=start_coords.dtype, device=device)
        start_coords[0, atom_is_lig] = center.unsqueeze(0) + noise
        print(
            f"  [covalent] seeded {n} ligand atoms near "
            f"({center_xyz[0]:.1f},{center_xyz[1]:.1f},{center_xyz[2]:.1f})"
        )
    return start_coords


def midpoint_ca(
    pdb_path: str,
    chain_a: str,
    res_a: int,
    chain_b: str,
    res_b: int,
) -> Tuple[float, float, float]:
    ca_a = _ca_coords_from_pdb(pdb_path, chain_a)
    ca_b = _ca_coords_from_pdb(pdb_path, chain_b)
    if res_a not in ca_a or res_b not in ca_b:
        # fallback: average of whatever exists
        pts = list(ca_a.values()) + list(ca_b.values())
        if not pts:
            return (0.0, 0.0, 0.0)
        xs = sum(p[0] for p in pts) / len(pts)
        ys = sum(p[1] for p in pts) / len(pts)
        zs = sum(p[2] for p in pts) / len(pts)
        return (xs, ys, zs)
    ax, ay, az = ca_a[res_a]
    bx, by, bz = ca_b[res_b]
    return ((ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0)
