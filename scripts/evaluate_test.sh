#!/bin/bash
# Evaluate the held-out test slice from saved method artifacts.
#
# This scores the same method keys as scores_train.json when that file exists;
# otherwise it falls back to scoring every method found in output manifests.

set -e

cd /home/user

mkdir -p scoring

mapfile -t MANIFESTS < <(find outputs -mindepth 2 -maxdepth 3 -name manifest.json | sort)
if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  echo "No output manifests found under outputs/." >&2
  exit 1
fi

python -m eval.test.evaluate \
  --manifest "${MANIFESTS[@]}" \
  --output scoring/scores_test.json \
  --match-train scoring/scores_train.json
