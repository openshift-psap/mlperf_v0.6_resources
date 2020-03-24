#!/bin/bash

python ./qa/testscript.py $1 --arch resnet50 -c fanin --mode inference --ngpus 1 --bs 1 2 4 8 16 32 64 128 --baseline qa/benchmark_baselines/RN50_pytorch_18.08-py3-stage_infer_fp32.json
