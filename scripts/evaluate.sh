#!/bin/bash
# Standard evaluation entry point (visible train slice).
#
# Self-contained: uses only eval/train/ (never imports method/).  Runs the
# private protocol with every player policy registered under outputs/ (or the
# policy manifests passed as arguments) on the public train instances and
# writes scoring/scores.json plus scoring/diagnostics.json.  The policy
# interface is described in data/protocol.py and briefing/evaluation.md.
#
# Usage:
#   bash scripts/evaluate.sh                              # all outputs/*/policy.json
#   bash scripts/evaluate.sh outputs/my_method/policy.json

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  POLICIES=("$@")
else
  mapfile -t POLICIES < <(find outputs -mindepth 2 -maxdepth 3 -name policy.json 2>/dev/null | sort)
fi
if [ "${#POLICIES[@]}" -eq 0 ]; then
  echo "No policy manifests found under outputs/ (expected outputs/<dir>/policy.json)." >&2
  exit 1
fi

python -m eval.train.evaluate \
  --policy "${POLICIES[@]}" \
  --output scoring/scores.json \
  --diagnostics scoring/diagnostics.json
