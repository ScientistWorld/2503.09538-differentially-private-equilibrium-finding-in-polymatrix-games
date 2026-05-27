# Scientist Constraints

## Problem Instances
Use the provided procedural polymatrix-game generator and the evaluation seeds used by the scripts. Do not replace the game family with an easier synthetic problem or evaluate on hand-selected games.

## Resource Scale
Use the same player counts, action-set size, graph topology, and number of random seeds as the submitted run unless the benchmark scripts are explicitly changed for everyone. Increasing player count or seed count can alter the scaling metrics and is resource scaling, not a method improvement.

## Privacy Accounting
Report the same Renyi-DP bound used by `scripts/evaluate.sh`. Do not improve exploitability by reducing noise, changing the privacy definition, changing adjacency, or dropping the privacy metric.

## Output Contract
Methods must write manifests and trajectory artifacts with the same fields expected by `eval/evaluate_runs.py`. The evaluator must not import method code.

## Compliance Checklist
- [ ] Uses the same procedural game generator, topology, seeds, action count, and player-count grid.
- [ ] Keeps the same privacy accounting, adjacency definition, and communication-observation setting.
- [ ] Writes artifacts in the expected manifest and trajectory format.
- [ ] Scores are produced by `bash scripts/evaluate.sh`, not by method-side self-reporting.
