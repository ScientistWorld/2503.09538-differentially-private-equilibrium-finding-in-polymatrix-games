#!/bin/bash
set -euo pipefail
cd /home/user
python -m redteam.audit_v2 comparator_baseline
