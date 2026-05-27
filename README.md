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

---

# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

How can a distributed group of agents compute an approximate equilibrium in a polymatrix game while keeping individual pairwise utility matrices private from an adversary who observes a limited number of communication channels?

## Why It Matters

Polymatrix games model networked multi-agent settings such as trading, security, and local coordination. In these settings, payoffs can reveal private preferences or valuations, so an equilibrium-finding protocol is useful only if it reaches a good equilibrium without exposing those payoffs.

## Success Criteria

A successful solution should output a low-exploitability coarse correlated equilibrium on random polymatrix games and should maintain a decreasing differential-privacy budget as the number of players grows. The evaluation must measure both equilibrium quality and privacy; improving only one side is insufficient.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper introduces a distributed differentially private algorithm for computing coarse correlated equilibria in polymatrix games, with guarantees that both exploitability and privacy budget can vanish as the number of players grows under limited adversarial channel access.

## Approach

At each iteration, every player broadcasts its mixed strategy plus isotropic Gaussian noise to its neighbors. Neighbors project the noisy broadcasts back to their simplices, compute local polymatrix gradients from these noisy strategies, and update by a Euclidean proximal projected-gradient step. The regularization strength is player-specific: it is proportional to the harmonic mean of graph degrees divided by the player's own degree, so low-degree players receive stronger stabilization.

## Main Claims

- Primary: under limited channel access, the algorithm can achieve low CCE exploitability and low Renyi-DP budget simultaneously as the number of players grows.
- Secondary: this simultaneous improvement holds in dense and sparse graph regimes, with different mechanisms governing the privacy bound.
- Negative result: if an adversary observes all channels, or if Euclidean distance to equilibrium is required instead of exploitability, vanishing accuracy and privacy cannot both be guaranteed.

---

# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

- **CCE exploitability**: average per-player regret of the empirical distribution over iterates. Lower is better.
- **Last-iterate Nash gap**: average immediate gain from unilateral deviation at the final strategy profile. Lower is better and is diagnostic only.
- **Renyi-DP privacy budget bound**: the single-channel average privacy upper bound for adjacent games that differ on one edge. Lower is better.
- **Scaling slopes**: log-log slopes of exploitability and privacy budget versus number of players. Negative slopes mean the metric improves as games get larger.

## Baselines and Targets

The reference baseline is a noisy online projected-gradient method adapted from private distributed optimization. The target behavior is lower exploitability than this baseline under the same noise/privacy schedule, while also showing a negative privacy-budget slope as player count increases.

## Evaluation Protocol

Generate random polymatrix games with fixed seeds, identical action-set size, and the same graph topology across compared methods. Each method writes a manifest plus saved game and trajectory artifacts. `scripts/evaluate.sh` reads those artifacts and computes scores without importing the method implementation.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 18:50] - method_runs
- Read the paper and identified Algorithm 1 as the reproducible core method.
- Implemented procedural polymatrix-game generation, Algorithm 1, a private noisy OMD baseline, and artifact-based evaluation.
- Ran a local CPU smoke test on small dense games; the method and evaluator execute end-to-end.

### [2026-05-27 19:02] - core_claim
- Completed a scheduled CPU reproduction job on dense Erdos-Renyi games with N in {64, 128, 256, 512} and five seeds.
- The proposed method's CCE exploitability decreased with player count (log-log slope -0.4119).
- The reproduced privacy budget bound also decreased with player count (log-log slope -0.0859), supporting the paper's simultaneous accuracy/privacy scaling claim at finite scale.

</details>


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
