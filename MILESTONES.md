# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-05-27 18:50] - method_runs
- Read the paper and identified Algorithm 1 as the reproducible core method.
- Implemented procedural polymatrix-game generation, Algorithm 1, a private noisy OMD baseline, and artifact-based evaluation.
- Ran a local CPU smoke test on small dense games; the method and evaluator execute end-to-end.

### [2026-05-27 19:02] - core_claim
- Completed a scheduled CPU reproduction job on dense Erdos-Renyi games with N in {64, 128, 256, 512} and five seeds.
- The proposed method's CCE exploitability decreased with player count (log-log slope -0.4119).
- The reproduced privacy budget bound also decreased with player count (log-log slope -0.0859), supporting the paper's simultaneous accuracy/privacy scaling claim at finite scale.

## Stop Justification
- Completed at milestone `core_claim`.
