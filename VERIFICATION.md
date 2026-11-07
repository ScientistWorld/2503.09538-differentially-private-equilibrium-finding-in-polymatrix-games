> **Superseded (hardening iterations):** this note describes the pre-hardening evaluator. The current design (player-policy interface, hidden instances, locality audit) and its audit are in `HARDENING.md`, `briefing/evaluation.md` and `scoring/`.

# Verification Report

## Initial Assessment

The workspace started at milestone `core_claim_plus`. The claimed milestone was broadly justified: the method implements the paper's private polymatrix-game equilibrium algorithm, existing artifacts covered dense Erdos-Renyi, dense clustered, and sparse kNN settings, and the train/test split evaluators were already present.

## Issues Found

- Minor: `briefing/problem.md` and `briefing/evaluation.md` still contained template instruction text.
- Minor: problem-side utilities and evaluators used theorem-specific helper names, which was unnecessary leakage in method-agnostic files.
- Minor: `scoring/EXPERIMENTS.md` documented an unscored sparse-random diagnostic even though the file should match `scores.json`.
- Minor: `briefing/evaluation.md` named the reference comparator's algorithm in a method-agnostic file.

## Fixes Applied

- Removed template text from method-agnostic briefing files.
- Renamed shared privacy/schedule helpers to neutral benchmark terminology and updated method, baseline, and evaluator call sites without changing the computation.
- Removed the unscored diagnostic entry from `scoring/EXPERIMENTS.md`; the sparse-random limitation remains documented in `PROGRESS.md`.
- Reworded the method-agnostic evaluation briefing to describe the comparator contract without naming the comparator algorithm.
- Ran CPU validation job `abee7929-170`, which regenerated standalone, train, and held-out test scores and passed `python validate.py --compare`.
- Accepted the later manifest-scoping changes by direct shell-script inspection and `bash -n`; they only restrict which manifests are passed to the already validated evaluator.
- Removed generated `jobs/` and `trajectory/` logs from the git index; `.gitignore` already excludes them and they remain local runtime evidence rather than shipped gym contents.

## Final Judgment

The workspace is ready to serve as a research gym. Problem/solution separation is clean, `eval/` reads saved method artifacts rather than importing method code, scoring includes both benefit and constraint metrics, and train/test evaluators have matching method keys with comparable behavior.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: finite-scale procedural games; sparse-random privacy scaling remains an unscored diagnostic gap.
