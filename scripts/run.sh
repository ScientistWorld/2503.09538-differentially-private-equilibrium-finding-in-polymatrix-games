#!/bin/bash
# Complete the audit's third independent reference rerun after fixing the
# candidate-wrapper import-cache collision. No evaluator arguments are changed.
set -euo pipefail
cd /home/user
python -m redteam.audit_v2 reference_seed2
python -m redteam.audit_v2 summarize
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('redteam/audit_v2_state/summary.json').read_text())
assert summary['seed_transcripts_identical']
assert summary['seed_rank_spread'] == 0
assert len(summary['results']) == 20
assert all(r['test_exit'] == 0 and r['evaluator_unchanged'] for r in summary['results'].values())
print('All 20 cases complete. Three actual reference seeds: identical transcripts and zero rank spread.')
PY
