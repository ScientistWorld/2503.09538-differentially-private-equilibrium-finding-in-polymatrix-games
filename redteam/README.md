Audit source and saved per-run scores. Each candidate has a `run.sh` that prepares
its persistent method artifacts, runs the visible training evaluator, and then
runs `bash scripts/evaluate_test.sh` without arguments in a separate process.
There are no surviving method processes or changes to evaluator code or Python.

Run the complete CPU audit with `python3 -m redteam.audit batch`. This preserves
and restores the starting `outputs/` and score files. Per-case scripts overwrite
the active `outputs/` manifests as a scientist's method would. They rebuild public
games and reference outputs when the ignored cache is absent. The original grid
is used for matched comparisons; the dropped-games shortcut deliberately changes
it. No shortcut reads `paper/`, `eval/test/`, or reference scores to generate its
outputs. Audit-only files under `_audit/` are used after generation to verify
evaluator identity and calculate rank gains.

All names prefixed `shortcut_` are shortcut probes, including rejected probes and
ties. The constant-uniform shortcut is also a legitimate simple alternative;
its score tests whether the task distinguishes private equilibrium research from
data-independent constant play.
