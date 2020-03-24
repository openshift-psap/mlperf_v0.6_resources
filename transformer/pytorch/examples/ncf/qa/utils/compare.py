import json
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--baseline', type=str, required=True)
parser.add_argument('--key', type=str, required=True)
parser.add_argument('--tolerance', type=float, default=0.9)

args = parser.parse_args()

input = json.load(open(args.input))
baseline = json.load(open(args.baseline))

input_value = input['run'][args.key]
baseline_value = baseline['run'][args.key]

print("input value: ", input_value)
print("baseline value: ", baseline_value)

# only 'higher is better'-type scenarios currently supported
if input_value < 0.9 * baseline_value:
    print('FAILED')
    sys.exit(1)
else:
    print('PASSED')
    sys.exit(0)
