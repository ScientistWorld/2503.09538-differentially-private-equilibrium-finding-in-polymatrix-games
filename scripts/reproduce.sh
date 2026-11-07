#!/bin/bash
# Reproduce the reference results end-to-end inside the container.
#
# Runs both baselines and the paper's method on every benchmark instance
# (3 topologies x 4 player counts x 12 seeds), then scores the visible slice
# (scoring/scores.json, scoring/scores_train.json) and the held-out slice
# (scoring/scores_test.json).  The fixed comparator values the evaluators
# compare against are committed in eval/train/ and eval/test/; rebuild them
# only if missing.

set -e

cd /home/user

for split in train test; do
  if [ ! -f "eval/$split/comparator_$split.json" ]; then
    python -m "eval.$split.evaluate" --build-comparator
  fi
done

rm -rf outputs/proposed outputs/baseline outputs/structure_free_baseline outputs/no_payoff_baseline

bash scripts/baseline.sh
bash scripts/method.sh

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
