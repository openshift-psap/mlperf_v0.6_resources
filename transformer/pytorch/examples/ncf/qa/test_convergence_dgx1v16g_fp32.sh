#!/bin/bash
set -e

HARDWARE="DGX_1V_16G"
PRECISION="fp32"

./run.sh --no-fp16

python qa/utils/compare.py --input nvlog.json --baseline ./qa/baseline_results/${HARDWARE}_${PRECISION}.json --key best_accuracy
