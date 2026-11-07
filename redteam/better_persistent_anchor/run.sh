#!/bin/bash
set -euo pipefail
cd /home/user
python3 -m redteam.audit run better_persistent_anchor
