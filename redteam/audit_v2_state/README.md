This audit targets commit `76a0754`, the current hardened gym. Older audit
directories and the old REDTEAM.md describe a different evaluator and are not
evidence for the current score.

Run `python -m redteam.audit_v2 batch` inside the gym container as a CPU job.
Each `redteam/audit_v2_<case>/run.sh` also independently installs its candidate,
runs it, then runs the unmodified train and no-argument test commands. Candidate
construction uses only public data/baseline code and the improve-mode reference
method. The paper, hidden comparator files, and hidden evaluator are never read
by candidate construction. Ranking reads the frozen score/rubric JSON only
after evaluation. All numeric trajectories are reproducible and ignored.

Reference reruns have process seeds 0, 1 and 2. The reference has no free RNG:
its only noise is fixed by the submission contract. Alternative noise-key
studies require changing evaluator arguments and are therefore excluded from
the reported three-seed spread for this audit.

Scores/diagnostics/result files are saved next to each candidate's run.sh.
The runtime harness restores the original outputs and scoring files after each
case, and verifies evaluator hashes. Its installed method wrapper is removed.
The temporary scheduler script will be restored before final delivery.

Job 0e3cdfc6-564 completed all 20 cases. Log review caught an ordinary Python
bytecode-cache collision: the seed-2 wrapper had the seed-1 wrapper's filename,
mtime and byte length, so Python executed seed 1 again. The harness now uses a
different module filename per case and checks reference invocation logs. A
follow-up job reruns seed 2; only the corrected three runs are used in the final
seed-spread claim. Other adjacent wrappers had different byte lengths and their
generation logs identify the intended case.
