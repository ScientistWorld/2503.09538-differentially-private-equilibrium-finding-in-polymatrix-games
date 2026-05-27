#!/bin/bash
# Score the existing final method artifacts on the visible train slice and
# held-out test slice. This job must not retrain or regenerate method outputs.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
