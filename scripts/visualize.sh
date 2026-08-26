#!/usr/bin/env bash
set -euo pipefail

# The evaluation already produces the per-token 3D trajectory browser and the
# polar usage chart. This compatibility entry runs that same evaluation.
bash scripts/eval.sh "$@"
