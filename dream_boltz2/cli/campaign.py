"""Run a design campaign in the order used by the manuscript .srp files.

Default path (plain binder, from DREAM.srp):

    dream -> screen -> wake -> screen -> LigandMPNN -> filter

Deferred covalent YAML (Trop-2 FSY, HER2 3-NT, PD-L1 CMN) adds a second wake:

    ... -> wake --covalent_deferred -> screen -> LigandMPNN -> filter -> mask G as X

Thrombin-style runs skip waking: pass --skip-wake.

Screen defaults match the shared .srp block: Rg <= 14, sheet <= 95, loop <= 40,
at least 3 hotspot contacts within 7 A, clash ratio <= 0.1. Waking uses a
warm start at sigma 20. The final LigandMPNN call writes 50 sequences at
temperature 0.2 and keeps 4 per backbone. Pass flags to override any of these.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

from dream_boltz2.config import parse_dream_config
from dream_boltz2.covalent.deferred import is_covalent_deferred

STAGE_DREAM = "dream"
STAGE_SCREEN = "screen"
STAGE_WAKE = "wake"
STAGE_SCREEN_WAKE = "screen_wake"
STAGE_WAKE_COVALENT = "wake_covalent"
STAGE_SCREEN_COVALENT = "screen_covalent"
STAGE_SEQUENCES = "sequences"
STAGE_FILTER = "filter"
STAGE_MASK = "mask"

ALL_STAGES = (
    STAGE_DREAM,
    STAGE_SCREEN,
    STAGE_WAKE,
    STAGE_SCREEN_WAKE,
    STAGE_WAKE_COVALENT,
    STAGE_SCREEN_COVALENT,
    STAGE_SEQUENCES,
    STAGE_FILTER,
    STAGE_MASK,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_checkpoint() -> Optional[str]:
    env = os.environ.get("BOLTZ_CHECKPOINT")
    if env:
        return env
    candidate = Path.home() / ".boltz" / "checkpoints" / "boltz2_conf.ckpt"
    if candidate.is_file():
        return str(candidate)
    return None


def _hotspots_from_config(receptor_chain: str, indices: Sequence[int]) -> str:
    return ",".join(f"{receptor_chain}{int(index)}" for index in indices)


def _run(cmd: List[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    root = str(_project_root())
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)


def _screen_cmd(pdb_dir: Path, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable, "-m", "dream_boltz2.cli.screen", str(pdb_dir),
        "--target-chain", args.target_chain,
        "--rg-filter",
        "--max-rg", str(args.max_rg),
        "--max-sheet", str(args.max_sheet),
        "--max-loop", str(args.max_loop),
        "--min-contacts", str(args.min_contacts),
        "--max-clash", str(args.max_clash),
        "--contact-distance", str(args.contact_distance),
    ]
    if args.hotspots:
        cmd.extend(["--hotspots", args.hotspots])
    if args.processes is not None:
        cmd.extend(["--processes", str(args.processes)])
    return cmd


def _wake_cmd(
    args: argparse.Namespace,
    input_dir: Path,
    out_dir: Path,
    covalent_deferred: bool,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "dream_boltz2.cli.wake",
        "--yaml_config", str(args.config),
        "--checkpoint", args.checkpoint,
        "--input_dir", str(input_dir),
        "--out_dir", str(out_dir),
        "--device", args.device,
        "--warm_start_sigma", str(args.sigma),
    ]
    if args.warm_start:
        cmd.append("--warm_start")
    if args.num_samples is not None:
        cmd.extend(["--num_samples", str(args.num_samples)])
    if args.wake_mpnn_seqs is not None:
        cmd.extend(["--mpnn_num_seqs", str(args.wake_mpnn_seqs)])
    if args.wake_mpnn_temperature is not None:
        cmd.extend(["--mpnn_temperature", str(args.wake_mpnn_temperature)])
    if covalent_deferred:
        cmd.append("--covalent_deferred")
    return cmd


def build_plan(args: argparse.Namespace) -> List[tuple]:
    """Return (stage_name, argv) pairs that will actually run."""
    out = Path(args.out)
    coarse_screen = out / "screen_results"
    wake_out = out / "relax_results"
    wake_pdbs = wake_out / "all_relaxed"
    wake_screen = wake_pdbs / "screen_results"
    cov_out = out / "relax_results_covalent"
    cov_pdbs = cov_out / "all_relaxed"
    cov_screen = cov_pdbs / "screen_results"

    if args.skip_wake:
        seq_input = coarse_screen
        seq_dir = out / "ligandmpnn_sequences"
    elif args.covalent:
        seq_input = cov_screen
        seq_dir = cov_pdbs / "ligandmpnn_sequences"
    else:
        seq_input = wake_screen
        seq_dir = wake_pdbs / "ligandmpnn_sequences"
    filtered = seq_dir / "filtered"

    plan: List[tuple] = []
    plan.append((
        STAGE_DREAM,
        [
            sys.executable, "-m", "dream_boltz2.cli.design",
            "--config", str(args.config),
            "--checkpoint", args.checkpoint,
            "--device", args.device,
            "--output_dir", str(out),
        ],
    ))
    plan.append((STAGE_SCREEN, _screen_cmd(out, args)))

    if not args.skip_wake:
        plan.append((STAGE_WAKE, _wake_cmd(args, coarse_screen, wake_out, False)))
        plan.append((STAGE_SCREEN_WAKE, _screen_cmd(wake_pdbs, args)))
        if args.covalent:
            plan.append((
                STAGE_WAKE_COVALENT,
                _wake_cmd(args, wake_screen, cov_out, True),
            ))
            plan.append((STAGE_SCREEN_COVALENT, _screen_cmd(cov_pdbs, args)))

    seq_cmd = [
        sys.executable, "-m", "dream_boltz2.cli.sequences",
        "--input_dir", str(seq_input),
        "--output_dir", str(seq_dir),
        "--binder_chain", args.binder_chain,
        "--receptor_chain", args.receptor_chain,
        "--yaml_config", str(args.config),
        "--num_sequences", str(args.num_sequences),
        "--temperature", str(args.temperature),
        "--batch_size", str(args.batch_size),
        "--model_type", args.model_type,
    ]
    if args.bias_aa_json:
        seq_cmd.extend(["--bias_AA_json", args.bias_aa_json])
    plan.append((STAGE_SEQUENCES, seq_cmd))
    plan.append((
        STAGE_FILTER,
        [
            sys.executable, "-m", "dream_boltz2.cli.filter_designs",
            "--outputs_dir", str(seq_dir),
            "--output_folder", str(filtered),
            "--top_n", str(args.top_n),
            "--remove_slash",
            "--keep_part", "binder",
            "--auto-detect-order",
            "--rename",
            "--name_map",
        ],
    ))
    if args.covalent and not args.skip_wake:
        plan.append((
            STAGE_MASK,
            [
                sys.executable, "-m", "dream_boltz2.cli.mask_covalent",
                "--fasta", str(filtered / "best_designs.fa"),
                "--attach_sites", str(cov_pdbs / "attach_sites.jsonl"),
                "--fixed_positions", str(seq_dir / "fixed_positions.jsonl"),
                "--name_mapping", str(filtered / "name_mapping.tsv"),
                "--binder_chain", args.binder_chain,
                "--strict",
            ],
        ))
    return plan


def _select(plan: List[tuple], start: str, end: str) -> List[tuple]:
    names = [name for name, _ in plan]
    if start not in names:
        raise SystemExit(
            f"--from {start} is not in this campaign. Stages: {', '.join(names)}"
        )
    if end not in names:
        raise SystemExit(
            f"--to {end} is not in this campaign. Stages: {', '.join(names)}"
        )
    i = names.index(start)
    j = names.index(end)
    if i > j:
        raise SystemExit(f"--from {start} is after --to {end}")
    return plan[i:j + 1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_ckpt = _default_checkpoint()
    parser = argparse.ArgumentParser(
        description="Run dream, screen, wake, and LigandMPNN as one campaign.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m dream_boltz2.cli.campaign --config examples/pd_l1.yaml\n"
            "  python -m dream_boltz2.cli.campaign --config examples/trop2_fluorosulfate.yaml "
            "--num-samples 2 --top-n 2\n"
            "  python -m dream_boltz2.cli.campaign --config examples/thrombin_sulfotyrosine.yaml "
            "--skip-wake\n"
            "  python -m dream_boltz2.cli.campaign --config examples/pd_l1.yaml "
            "--from wake --to screen_wake\n"
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="Campaign YAML.")
    parser.add_argument(
        "--checkpoint",
        default=default_ckpt,
        help="Boltz-2 checkpoint. Default: $BOLTZ_CHECKPOINT, else ~/.boltz/checkpoints/boltz2_conf.ckpt.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: dream.output.dir in the YAML.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--hotspots",
        default=None,
        help="Screen hotspots, e.g. A41,A39,A98. Default: receptor chain plus dream.hotspot_indices.",
    )
    parser.add_argument("--target-chain", default=None, help="Binder chain for screening. Default: YAML binder chain.")
    parser.add_argument("--binder-chain", default=None, help="Binder chain for LigandMPNN. Default: YAML binder chain.")
    parser.add_argument("--receptor-chain", default=None, help="Receptor chain. Default: YAML receptor chain.")
    parser.add_argument("--max-rg", type=float, default=14.0)
    parser.add_argument("--max-sheet", type=float, default=95.0)
    parser.add_argument("--max-loop", type=float, default=40.0)
    parser.add_argument("--min-contacts", type=int, default=3)
    parser.add_argument("--max-clash", type=float, default=0.1)
    parser.add_argument("--contact-distance", type=float, default=7.0)
    parser.add_argument("--processes", type=int, default=None, help="Screen worker count.")
    parser.add_argument("--sigma", type=float, default=20.0, help="Warm-start noise sigma.")
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Wake from noise instead of the backbone coordinates.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Structures per backbone during waking. Default: wake's own default (5).",
    )
    parser.add_argument("--wake-mpnn-seqs", type=int, default=None, help="Sequences sampled inside each wake.")
    parser.add_argument("--wake-mpnn-temperature", type=float, default=None)
    parser.add_argument("--num-sequences", type=int, default=50, help="Final LigandMPNN sequences per structure.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=4, help="Sequences kept per backbone.")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "ligand_mpnn", "soluble_mpnn", "protein_mpnn"],
    )
    parser.add_argument("--bias-aa-json", default=None, help="Optional LigandMPNN amino-acid bias JSON.")
    parser.add_argument(
        "--skip-wake",
        action="store_true",
        help="Screen the dreamed backbones, then design sequences. No relaxation.",
    )
    covalent = parser.add_mutually_exclusive_group()
    covalent.add_argument(
        "--covalent",
        action="store_true",
        help="Force the second covalent wake even if the YAML does not set covalent_mode.",
    )
    covalent.add_argument(
        "--no-covalent",
        action="store_true",
        help="Do not run the second wake, even if the YAML sets covalent_mode: deferred.",
    )
    parser.add_argument("--from", dest="start", default=STAGE_DREAM, choices=ALL_STAGES)
    parser.add_argument(
        "--to",
        dest="end",
        default=None,
        choices=ALL_STAGES,
        help="Last stage to run. Default: the last stage of this campaign.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and do not run them.")
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2
    if not args.checkpoint:
        print(
            "Pass --checkpoint, or set BOLTZ_CHECKPOINT. "
            "No ~/.boltz/checkpoints/boltz2_conf.ckpt was found.",
            file=sys.stderr,
        )
        return 2
    if not args.dry_run and not Path(args.checkpoint).is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    config = parse_dream_config(args.config)
    raw = yaml.safe_load(args.config.read_text())
    receptor = (args.receptor_chain or (config.receptor_chains[:1] or ["A"])[0])
    binder = (args.binder_chain or (config.binder_chains[:1] or ["B"])[0])
    args.receptor_chain = receptor
    args.binder_chain = binder
    args.target_chain = args.target_chain or binder
    if not args.hotspots and config.hotspot_indices:
        args.hotspots = _hotspots_from_config(receptor, config.hotspot_indices)
    args.out = args.out or config.output.dir
    args.warm_start = not args.no_warm_start
    if args.no_covalent:
        args.covalent = False
    elif not args.covalent:
        args.covalent = is_covalent_deferred(raw)
    args.config = args.config.resolve()

    plan = build_plan(args)
    plan = _select(plan, args.start, args.end or plan[-1][0])
    print(f"config: {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"output: {args.out}")
    print(f"hotspots: {args.hotspots or '(none)'}")
    print(f"covalent second wake: {args.covalent and not args.skip_wake}")
    print("stages: " + " -> ".join(name for name, _ in plan))
    try:
        for _name, cmd in plan:
            _run(cmd, args.dry_run)
    except subprocess.CalledProcessError as exc:
        print(f"Stage failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
