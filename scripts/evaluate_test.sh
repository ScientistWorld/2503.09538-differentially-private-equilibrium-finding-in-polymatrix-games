#!/bin/bash
# Evaluate the held-out test slice for every method found under outputs/.
#
# Runs with no arguments: discovers all output manifests, scores them with
# eval/test/ only (fixed games, operating point, noise stream and comparator
# values live there), and writes scoring/scores_test.json and
# scoring/diagnostics_test.json.  It never retrains or reruns a method.

set -e

cd /home/user

mkdir -p scoring

mapfile -t MANIFESTS < <(find outputs -mindepth 2 -maxdepth 4 -name manifest.json 2>/dev/null | sort)
if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  echo "No output manifests found under outputs/." >&2
  exit 1
fi

python -m eval.test.evaluate \
  --manifest "${MANIFESTS[@]}" \
  --output scoring/scores_test.json \
  --diagnostics scoring/diagnostics_test.json
