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
The temporary scheduler script was restored before final delivery. Job logs
and metadata are preserved in this directory's jobs/ subdirectory; per-case
process logs are saved as .txt beside each run.sh.

Job 0e3cdfc6-564 completed all 20 cases. Log review caught an ordinary Python
bytecode-cache collision: the seed-2 wrapper had the seed-1 wrapper's filename,
mtime and byte length, so Python executed seed 1 again. The harness now uses a
different module filename per case and checks reference invocation logs.
Follow-up job `cfe3c415-076` reran seed 2 and exited 0; only the corrected
three runs are used in the final seed-spread claim. All three trajectory hash
lists and score files match exactly. Other adjacent wrappers had different byte lengths and their
generation logs identify the intended case.

Final cleanup restored the original scripts/run.sh and removed this audit's
new job files from jobs/ after preserving them here. Platform-masked README.md
and manifest.json and read-only trajectory/ cannot be restored in the worktree;
their original git blobs were restored in the index. These mounts were present
before the audit and were not modified. The final committed benchmark tree
should differ only in redteam/, REDTEAM.md and the required action.yaml.
