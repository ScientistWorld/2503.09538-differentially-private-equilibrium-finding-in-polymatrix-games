#!/bin/bash
# Standard evaluation entry point (visible train slice).
#
# Self-contained: uses only eval/train/ (never method/).  Scores every method
# whose manifest is found under outputs/ (or the manifests passed as
# arguments) and writes scoring/scores.json plus scoring/diagnostics.json.
# Each manifest lists (topology, n_players, seed, trajectory_path) runs; see
# data/submission.py for the artifact contract.
#
# Usage:
#   bash scripts/evaluate.sh                                  # all manifests under outputs/
#   bash scripts/evaluate.sh outputs/my_method/*/manifest.json

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  MANIFESTS=("$@")
else
  mapfile -t MANIFESTS < <(find outputs -mindepth 2 -maxdepth 4 -name manifest.json 2>/dev/null | sort)
fi
if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  echo "No output manifests found under outputs/." >&2
  exit 1
fi

python -m eval.train.evaluate \
  --manifest "${MANIFESTS[@]}" \
  --output scoring/scores.json \
  --diagnostics scoring/diagnostics.json
