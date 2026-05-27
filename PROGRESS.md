# Progress

## What Works

Algorithm 1 runs end-to-end on procedurally generated polymatrix games. The implementation includes Gaussian noisy strategy broadcasts, projection of noisy broadcasts to the simplex, degree-adaptive regularization from the harmonic mean of degrees, and the projected proximal update.

The reusable evaluation pipeline reads saved manifests, games, and trajectories from `outputs/` and writes `scoring/scores.json` without importing from `method/`.

## Results

The scheduled reproduction job completed successfully on dense Erdos-Renyi games with player counts 64, 128, 256, and 512 across five seeds.

- Proposed method CCE exploitability log-log slope: `-0.4119` (lower is better; negative supports decreasing exploitability).
- Proposed privacy bound log-log slope: `-0.0859` (lower is better; negative supports improving privacy with player count).
- Largest evaluated game CCE exploitability: `0.0535`.
- The noisy OMD baseline was close in this finite-scale setting and slightly better on mean exploitability ratio, so the reproduction supports the simultaneous accuracy/privacy scaling claim but not the full empirical baseline gap reported in the paper.

## Remaining

The next submitted job expands beyond `core_claim` to the paper's other graph topologies: dense clustered graphs, sparse random graphs, and sparse k-nearest-neighbor graphs. A closer implementation of the Huang et al. adapted baseline would still be needed to reproduce the paper's baseline separation more faithfully.

## Issues

The paper reports trends mostly in figures rather than numeric tables. `reference.json` therefore scores the qualitative trend indicators reported in the paper, while this file records the raw reproduced slopes as diagnostics. In this finite reproduction, the proposed method and baseline have similar exploitability, while the central simultaneous decrease of exploitability and privacy bound is reproduced.

Validation note, 2026-05-27 19:04 UTC: the score contract was changed from non-paper-reported numeric slopes to binary trend indicators so `scores.json` compares like-for-like against claims actually reported by the paper. The completed run's raw slopes remain listed above.

Update, 2026-05-27 19:25 UTC: prepared a multi-topology reproduction job. Dense graphs are scored for decreasing exploitability and privacy budget; sparse graphs are scored for non-increasing exploitability and decreasing privacy budget, matching Appendix D's qualitative claims.

Audit update, 2026-05-27 20:05 UTC: continuing agent reviewed `method/`, `data/`, and `eval/`. The method is a direct implementation of Algorithm 1 rather than a surrogate. One scoring issue was found before rerunning: raw diagnostic slopes were being written into `scores.json` even though they are not declared metrics in `reference.json`. This was fixed by writing diagnostics to `scoring/diagnostics.json` and keeping `scores.json` limited to declared benchmark metrics.

## Deviations from Paper

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

     The test: could someone reading your code identify which algorithm from
     the paper you implemented? If not, you replaced the method. -->

- The paper ran large PyTorch experiments on 2 x NVIDIA H200 GPUs. This reproduction uses NumPy and a smaller finite player-count grid to fit the available budget while preserving the same algorithmic steps and evaluation quantities.
