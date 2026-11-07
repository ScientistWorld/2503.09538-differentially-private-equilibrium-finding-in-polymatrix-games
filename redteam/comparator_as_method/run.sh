#!/bin/bash
set -euo pipefail
cd /home/user
python3 -m redteam.audit run comparator_as_method
