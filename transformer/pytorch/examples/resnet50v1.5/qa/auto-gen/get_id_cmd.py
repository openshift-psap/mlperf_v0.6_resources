import json
import sys

j = json.load(open(sys.argv[1], 'r'))
print(j['id'])
