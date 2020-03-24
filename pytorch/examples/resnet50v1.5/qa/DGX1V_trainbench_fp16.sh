#!/bin/bash

python ./qa/testscript.py $1 --arch resnet50 -c fanin --mode training --bench-warmup 2 --bench-iterations 1000 --ngpus 1 4 8 --bs 64 128 256 --fp16 --baseline qa/benchmark_baselines/RN50_pytorch_18.08-py3-stage_train_fp16.json
