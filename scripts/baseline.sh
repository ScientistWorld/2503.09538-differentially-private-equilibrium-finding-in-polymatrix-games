#!/bin/bash
# Register the baselines as player policies.
#
#   structure_free_baseline -- prior-work-style private learner whose step size
#                              is calibrated to the same privacy budget with an
#                              accounting that ignores the graph structure
#                              (the comparator whose fixed values live in eval/).
#   no_payoff_baseline      -- follows the noisy protocol but never uses payoffs.
#
# Usage:
#   bash scripts/baseline.sh            # both
#   bash scripts/baseline.sh structure_free

set -e

cd /home/user

WHICH="${1:-all}"
if [ "$WHICH" = "all" ] || [ "$WHICH" = "structure_free" ]; then
  python -m baseline.structure_free --output-dir outputs/structure_free_baseline
fi
if [ "$WHICH" = "all" ] || [ "$WHICH" = "no_payoff" ]; then
  python -m baseline.no_payoff --output-dir outputs/no_payoff_baseline
fi
