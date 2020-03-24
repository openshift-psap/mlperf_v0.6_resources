#!/bin/bash
set -e

HARDWARE="DGX_2"
PRECISION="fp32"

WORLD_SIZE=16 ./run.sh --no-fp16

python qa/utils/compare.py --input nvlog.json --baseline ./qa/baseline_results/${HARDWARE}_${PRECISION}.json --key best_train_throughput
