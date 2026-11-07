#!/bin/bash
# Run matched scientific controls and persisted-artifact shortcut probes.

set -euo pipefail

cd /home/user

python3 -m redteam.audit batch
