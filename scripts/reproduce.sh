#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

rm -rf outputs/proposed outputs/baseline

export N_VALUES="${N_VALUES:-64 128 256 512}"
export ACTIONS="${ACTIONS:-5}"
export P_EDGE="${P_EDGE:-0.12}"
export SPARSE_DEGREE="${SPARSE_DEGREE:-4}"
export SEEDS="${SEEDS:-0 1 2 3 4}"
export ALPHA="${ALPHA:-2.0}"
export AUTO_EVALUATE=0

for TOPOLOGY in dense_er dense_cluster sparse_random sparse_knn; do
  export TOPOLOGY
  export OUTPUT_SUFFIX="$TOPOLOGY"
  bash scripts/baseline.sh
  bash scripts/method.sh
done

bash scripts/evaluate.sh
