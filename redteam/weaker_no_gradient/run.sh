#!/bin/bash
set -euo pipefail
cd /home/user
python3 -m redteam.audit run weaker_no_gradient
