#!/bin/bash
# Run the paper's proposed method (Algorithm 1) on every benchmark instance.
#
# Writes outputs/proposed/<topology>/manifest.json plus one .npz per instance.
# Optional environment overrides: TOPOLOGIES, N_VALUES, SEEDS.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

ARGS=()
[ -n "${TOPOLOGIES:-}" ] && ARGS+=(--topologies ${TOPOLOGIES})
[ -n "${N_VALUES:-}" ] && ARGS+=(--n-values ${N_VALUES})
[ -n "${SEEDS:-}" ] && ARGS+=(--seeds ${SEEDS})

python -m method.dp_cce --output-dir outputs/proposed "${ARGS[@]}"
