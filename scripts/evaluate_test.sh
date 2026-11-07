#!/bin/bash
# Evaluate the hidden test slice for every method found under outputs/.
#
# Runs with no arguments: discovers every player policy registered under
# outputs/ (outputs/<dir>/policy.json), runs the private protocol with each
# one on the hidden test instances (games, noise, comparator values and audit
# draw live only in eval/test/), verifies and scores the transcripts, and
# writes scoring/scores_test.json and scoring/diagnostics_test.json.
# Nothing is trained; submitted code runs only in child processes.

set -e

cd /home/user

mkdir -p scoring

mapfile -t POLICIES < <(find outputs -mindepth 2 -maxdepth 3 -name policy.json 2>/dev/null | sort)
if [ "${#POLICIES[@]}" -eq 0 ]; then
  echo "No policy manifests found under outputs/ (expected outputs/<dir>/policy.json)." >&2
  exit 1
fi

python -m eval.test.evaluate \
  --policy "${POLICIES[@]}" \
  --output scoring/scores_test.json \
  --diagnostics scoring/diagnostics_test.json
