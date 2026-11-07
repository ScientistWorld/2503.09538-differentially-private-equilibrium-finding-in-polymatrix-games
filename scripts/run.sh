#!/bin/bash
# Current scoring audit: quality controls, three actual reference reruns,
# private-information/global-objective shortcuts, and artifact/degenerate checks.
# All computation runs in this CPU job, not on the login node.
set -euo pipefail
cd /home/user
python -m redteam.audit_v2 batch
