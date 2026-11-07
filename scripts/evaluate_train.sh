#!/bin/bash
# Evaluate the visible train slice for every method found under outputs/.
#
# Discovers all output manifests (no arguments, no method names), scores them
# with eval/train/ only, and writes scoring/scores_train.json and
# scoring/diagnostics_train.json.

set -e

cd /home/user

mkdir -p scoring

mapfile -t MANIFESTS < <(find outputs -mindepth 2 -maxdepth 4 -name manifest.json 2>/dev/null | sort)
if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  echo "No output manifests found under outputs/." >&2
  exit 1
fi

python -m eval.train.evaluate \
  --manifest "${MANIFESTS[@]}" \
  --output scoring/scores_train.json \
  --diagnostics scoring/diagnostics_train.json
