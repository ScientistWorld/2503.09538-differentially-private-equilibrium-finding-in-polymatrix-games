#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

rm -rf outputs/proposed outputs/baseline

export TOPOLOGY=dense_er
export N_VALUES="64 128 256 512"
export ACTIONS=5
export P_EDGE=0.12
export SEEDS="0 1 2 3 4"
export ALPHA=2.0

bash scripts/baseline.sh
bash scripts/method.sh
python validate.py --compare
