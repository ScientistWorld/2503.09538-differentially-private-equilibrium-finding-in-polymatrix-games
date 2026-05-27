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
