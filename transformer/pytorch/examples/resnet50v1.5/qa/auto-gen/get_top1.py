import json
import sys

with open(sys.argv[1], 'r') as f:
    j = json.load(f)
    print(j['run']['epochs'], j['run']['lr'], j['run']['momentum'], j['run']['weight_decay'], j['run']['warmup'], max(j['epoch']['val.top1']))


