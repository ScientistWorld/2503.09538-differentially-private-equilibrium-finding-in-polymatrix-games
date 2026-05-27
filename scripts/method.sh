#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

OUT_DIR="outputs/proposed/${OUTPUT_SUFFIX:-${TOPOLOGY:-dense_er}}"

python -m method.dp_cce \
  --output-dir "$OUT_DIR" \
  --topology "${TOPOLOGY:-dense_er}" \
  --n-values ${N_VALUES:-64 128 256} \
  --actions "${ACTIONS:-5}" \
  --p "${P_EDGE:-0.12}" \
  --c "${SPARSE_DEGREE:-4}" \
  --seeds ${SEEDS:-0 1 2} \
  --alpha "${ALPHA:-2.0}"

if [ "${AUTO_EVALUATE:-1}" = "1" ]; then
  MANIFESTS=("$OUT_DIR/manifest.json")
  MATCHED_BASELINE="outputs/baseline/${OUTPUT_SUFFIX:-${TOPOLOGY:-dense_er}}/manifest.json"
  if [ -f "$MATCHED_BASELINE" ]; then
    MANIFESTS+=("$MATCHED_BASELINE")
  fi
  bash scripts/evaluate.sh "${MANIFESTS[@]}"
fi
