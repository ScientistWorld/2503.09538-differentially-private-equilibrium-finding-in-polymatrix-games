#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

OUT_DIR="outputs/baseline/${OUTPUT_SUFFIX:-${TOPOLOGY:-dense_er}}"

python -m baseline.noisy_omd \
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
  MATCHED_PROPOSED="outputs/proposed/${OUTPUT_SUFFIX:-${TOPOLOGY:-dense_er}}/manifest.json"
  if [ -f "$MATCHED_PROPOSED" ]; then
    MANIFESTS+=("$MATCHED_PROPOSED")
  fi
  bash scripts/evaluate.sh "${MANIFESTS[@]}"
fi
