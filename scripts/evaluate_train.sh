#!/bin/bash
# Evaluate the visible train slice from saved method artifacts.
#
# Mirrors scripts/evaluate.sh: discover output manifests, score them without
# importing method/, and write a JSON scores file under scoring/.

set -e

cd /home/user

mkdir -p scoring

mapfile -t MANIFESTS < <(find outputs -mindepth 2 -maxdepth 3 -name manifest.json | sort)
if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  echo "No output manifests found under outputs/." >&2
  exit 1
fi

python -m eval.train.evaluate \
  --manifest "${MANIFESTS[@]}" \
  --output scoring/scores_train.json
