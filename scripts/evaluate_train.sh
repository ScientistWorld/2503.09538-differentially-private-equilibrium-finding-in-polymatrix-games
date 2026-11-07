#!/bin/bash
# Evaluate the visible train slice for every method found under outputs/.
#
# Discovers every player policy registered under outputs/ (no arguments, no
# method names), runs the private protocol with each one on the public train
# instances with eval/train/ only, and writes scoring/scores_train.json and
# scoring/diagnostics_train.json.

set -e

cd /home/user

mkdir -p scoring

mapfile -t POLICIES < <(find outputs -mindepth 2 -maxdepth 3 -name policy.json 2>/dev/null | sort)
if [ "${#POLICIES[@]}" -eq 0 ]; then
  echo "No policy manifests found under outputs/ (expected outputs/<dir>/policy.json)." >&2
  exit 1
fi

python -m eval.train.evaluate \
  --policy "${POLICIES[@]}" \
  --output scoring/scores_train.json \
  --diagnostics scoring/diagnostics_train.json
