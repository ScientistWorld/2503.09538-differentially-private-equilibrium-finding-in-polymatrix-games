#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./evaluate.sh <method_and_run_name>                             # evaluate with defaults
#   ./evaluate.sh <method_and_run_name> checkpoints/epoch_10.pt     # evaluate a specific checkpoint
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  MANIFESTS=("$@")
else
  mapfile -t MANIFESTS < <(find outputs -mindepth 2 -maxdepth 3 -name manifest.json | sort)
fi
if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  echo "No output manifests found under outputs/." >&2
  exit 1
fi

python -m eval.evaluate_runs \
  --manifest "${MANIFESTS[@]}" \
  --output scoring/scores.json
