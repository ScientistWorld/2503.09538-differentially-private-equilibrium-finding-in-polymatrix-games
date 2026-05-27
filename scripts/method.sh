#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.dp_cce \
  --output-dir outputs/proposed \
  --topology "${TOPOLOGY:-dense_er}" \
  --n-values ${N_VALUES:-64 128 256} \
  --actions "${ACTIONS:-5}" \
  --p "${P_EDGE:-0.12}" \
  --c "${SPARSE_DEGREE:-4}" \
  --seeds ${SEEDS:-0 1 2} \
  --alpha "${ALPHA:-2.0}"

bash scripts/evaluate.sh
