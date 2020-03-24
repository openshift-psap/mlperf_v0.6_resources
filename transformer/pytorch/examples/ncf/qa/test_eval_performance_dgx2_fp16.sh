#!/bin/bash
set -e

HARDWARE="DGX_2"
PRECISION="fp16"

WORLD_SIZE=16 ./run.sh

python qa/utils/compare.py --input nvlog.json --baseline ./qa/baseline_results/${HARDWARE}_${PRECISION}.json --key best_eval_throughput
