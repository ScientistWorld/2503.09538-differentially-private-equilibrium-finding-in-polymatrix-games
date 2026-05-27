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

export N_VALUES="64 128 256 512"
export ACTIONS=5
export P_EDGE=0.12
export SPARSE_DEGREE=4
export SEEDS="0 1 2 3 4"
export ALPHA=2.0
export AUTO_EVALUATE=0

for TOPOLOGY in dense_er dense_cluster sparse_random sparse_knn; do
  export TOPOLOGY
  export OUTPUT_SUFFIX="$TOPOLOGY"
  bash scripts/baseline.sh
  bash scripts/method.sh
done

bash scripts/evaluate.sh
python validate.py --compare
