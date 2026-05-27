# Overview

- **Paper ID:** 2503.09538
- **Title:** Differentially Private Equilibrium Finding in Polymatrix Games
- **Domain:** Game Theory / Differential Privacy / Distributed Optimization
- **TL;DR:** The paper studies when private distributed equilibrium computation is possible and proposes a noisy, degree-regularized CCE algorithm whose accuracy and privacy improve with player count.

## Short Summary

The problem is to compute approximate equilibria in polymatrix games without revealing sensitive pairwise utility matrices through the communication process. The paper proves impossibility results for overly strong observation or accuracy requirements, then gives a positive result for exploitability under limited channel access. Its algorithm adds Gaussian noise to strategy broadcasts and uses adaptive regularization based on graph degree. Experiments on random dense and sparse graph topologies evaluate exploitability and privacy-budget scaling.

## Key Results

- The proposed algorithm targets coarse correlated equilibrium exploitability, not Euclidean distance to the equilibrium set.
- The privacy metric is a Renyi-DP bound for adjacent games differing on one edge.
- The numerical claim is scaling: exploitability and privacy budget improve with player count in dense graphs.
