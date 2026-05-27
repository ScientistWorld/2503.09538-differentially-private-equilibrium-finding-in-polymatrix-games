# Evaluation

## Metrics

- **CCE exploitability**: average per-player fixed-deviation regret of the empirical distribution over iterates. Lower is better.
- **Last-iterate Nash gap**: average immediate gain from unilateral deviation at the final strategy profile. Lower is better and is diagnostic only.
- **Renyi-DP privacy budget bound**: the single-channel average privacy upper bound for adjacent games that differ on one edge, using the dense or sparse regime branch analyzed for the evaluated graph family. Lower is better.
- **Scaling slopes**: log-log slopes of exploitability and privacy budget versus number of players. Negative slopes mean the metric improves as games get larger.

## Baselines and Targets

Each submitted method is compared with a fixed reference comparator through the same artifact interface. The target behavior is competitive or lower exploitability than the comparator on matched games while also showing the claimed exploitability/privacy scaling trends as player count increases.

## Evaluation Protocol

Generate random polymatrix games with fixed seeds, identical action-set size, and the same graph topology across compared methods. Each method writes a manifest plus saved game and trajectory artifacts. `scripts/evaluate.sh` reads those artifacts and computes scores without importing the method implementation.
