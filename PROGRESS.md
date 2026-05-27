# Progress

## What Works

Algorithm 1 runs end-to-end on procedurally generated polymatrix games. The implementation includes Gaussian noisy strategy broadcasts, projection of noisy broadcasts to the simplex, degree-adaptive regularization from the harmonic mean of degrees, and the projected proximal update.

The reusable evaluation pipeline reads saved manifests, games, and trajectories from `outputs/` and writes `scoring/scores.json` without importing from `method/`.

## Results

A small local smoke test confirmed execution and scoring, but it is not sufficient to claim the central scaling result. A larger scheduler run is prepared for dense Erdos-Renyi games with player counts 64, 128, 256, and 512.

## Remaining

Run the scheduled reproduction job, inspect `scoring/scores.json`, and determine whether the dense-graph exploitability and privacy scaling support the core claim. If the result is noisy, adjust finite-scale player counts or seeds while preserving the paper's procedure.

## Issues

The paper reports trends mostly in figures rather than numeric tables, so `reference.json` encodes the reported theoretical/qualitative dense scaling targets rather than exact digitized values.

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
