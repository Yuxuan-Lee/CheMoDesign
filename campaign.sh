#!/usr/bin/env bash
# Run one campaign: dream, screen, wake, sequence design.
# Arguments are forwarded to dream_boltz2.cli.campaign.
# Example:
#   ./campaign.sh --config examples/pd_l1.yaml --checkpoint /path/to/boltz2_conf.ckpt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-python}"
exec "${PYTHON}" -m dream_boltz2.cli.campaign "$@"
