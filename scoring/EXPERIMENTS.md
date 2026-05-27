# Experiment Overview

## dense_er_scaling

Tests the central scaling claim on random dense Erdos-Renyi polymatrix games: equilibrium quality and privacy budget should both improve as the number of players increases.

## dense_cluster_scaling

Tests whether the dense-graph behavior also appears on the clustered dense topology used in the paper's additional experiments.

## sparse_knn_scaling

Tests the sparse geometric nearest-neighbor topology from the paper's additional experiments under the same sparse-graph success criteria.

## sparse_random_scaling Diagnostic

Sparse random graphs from Figure 6 are generated during reproduction and emitted to `scoring/diagnostics.json`. They are intentionally not scored because the finite-scale grid used here did not reproduce the paper's decreasing privacy-budget trend.
