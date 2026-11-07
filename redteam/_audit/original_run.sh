#!/bin/bash
# Validate the packaged evaluators against the existing final method artifacts.
# This job must not retrain or regenerate method outputs.

set -euo pipefail

cd /home/user

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare
