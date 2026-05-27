# Experiment Overview

## dense_er_scaling

Tests the central scaling claim on random dense Erdos-Renyi polymatrix games: equilibrium quality and privacy budget should both improve as the number of players increases.

## dense_cluster_scaling

Tests whether the dense-graph behavior also appears on the clustered dense topology used in the paper's additional experiments.

## sparse_random_scaling

Tests the sparse random graph setting. The expected behavior is improving privacy with player count while equilibrium quality stays roughly stable rather than worsening.

## sparse_knn_scaling

Tests the sparse geometric nearest-neighbor topology from the paper's additional experiments under the same sparse-graph success criteria.
