#!/bin/bash
set -euo pipefail
cd /home/user
python3 -m redteam.audit run shortcut_drop_small_games
