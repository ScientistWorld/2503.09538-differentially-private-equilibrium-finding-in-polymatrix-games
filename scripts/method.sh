#!/bin/bash
# Register the paper's proposed method (Algorithm 1) as a player policy.
#
# Writes outputs/proposed/policy.json (method "our_method", module
# method.dp_cce).  The evaluators run the private protocol with it on their own
# instances; nothing is trained.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.dp_cce --output-dir outputs/proposed
