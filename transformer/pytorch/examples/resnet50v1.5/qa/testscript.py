import os
import argparse
import json
import sys
import subprocess
import statistics as stat


from collections import OrderedDict

PERF_THR = 0.9

parser = argparse.ArgumentParser(description='PyTorch Benchmark Tests')

parser.add_argument('--bs', default=[1], type=int, nargs='+')
parser.add_argument('--ngpus', default=[1], type=int, nargs='+')

parser.add_argument('--mode', default='training', choices=['training', 'inference'],
                    help='benchmark training or inference (default: training)')
parser.add_argument('--bench-iterations', type=int, default=100, metavar='N',
                    help='Run N iterations while benchmarking (ignored when training and validation)')
parser.add_argument('--bench-warmup', type=int, default=3, metavar='N',
                    help='Number of warmup iterations for benchmarking')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--data-backends', default=['pytorch'], type=str, nargs='+')

parser.add_argument('--amp', action='store_true',
                    help='Run model amp mode.')

parser.add_argument('--baseline', type=str, default=None, metavar='FILE',
                    help='path to the file with baselines')

parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 5)')

parser.add_argument('--workspace', default="./", type=str, metavar='<PATH>')

parser.add_argument('data', default="/data/imagenet", type=str, metavar='<PATH>',
                    help='path to the dataset')

args, rest = parser.parse_known_args()

print(rest)

command = "{{}} main.py -p 1 -j {workers} {mode} {fp16} --workspace {workspace} --epochs {epochs} --prof {iters} {{}} --no-checkpoints {data} " + " ".join(rest)

if args.fp16:
    fp16 = "--fp16"
elif args.amp:
    fp16 = "--amp"
else:
    fp16 = ""

mode = "--evaluate" if args.mode == "inference" else "--training-only"

command = command.format(workers = args.workers,
                         mode=mode,
                         workspace=args.workspace,
                         fp16=fp16,
                         epochs = 2,
                         iters = args.bench_iterations,
                         data = args.data)

def benchmark(mode, ngpus, bspgpus, databackends, command, skip_first_n, metrics, workspace):
    sgpu = str(sys.executable)
    mgpu = "{} multiproc.py --nproc_per_node {{}}".format(str(sys.executable))

    table = { k: [] for k in metrics }

    od = OrderedDict
    dbr = lambda: [ (db, -1) for db in databackends ]
    bsr = lambda: [ (bs, od(dbr())) for bs in bspgpus ]
    ngr = lambda: [ (ngpu, od(bsr())) for ngpu in ngpus ]
    table = od( [ (m, od(ngr())) for m in metrics ] )

    for ngpu in ngpus:
        for bspgpu in bspgpus:
            for databackend in databackends:
                rfile = os.path.join(workspace, "./raport_{}GPU_{}BS_{}_{}bench.json".format(ngpu, bspgpu, databackend, mode))
                args = (" -b {} --data-backend {}".format(bspgpu, databackend)) + (" --raport-file " + rfile)

                cmd = command.format(sgpu if ngpu == 1 else mgpu.format(ngpu), args)

                print(cmd.split())
                exit_code = subprocess.call(cmd.split())

                if exit_code != 0:
                    print("CMD: \"{}\" exited with status {}".format("".join(cmd), exit_code))
                    for m in metrics:
                        table[m][ngpu][bspgpu][databackend] = -1.0

                else:
                    print("Job ended sucessfully")

                    raport = json.load(open(rfile, 'r'))

                    for m in metrics:
                        print(m)
                        print(raport['iter'][m])
                        table[m][ngpu][bspgpu][databackend] = stat.mean(raport['iter'][m][-2][skip_first_n:])

    def format_float(f):
        return "{:>10.1f}".format(f)

    def format_int(i):
        return "{:>8}".format(i)

    for m in metrics:
        if len(databackends) > 1:
            columns = [ "{} {}".format(format_int(ngpu), "-".join(map(lambda x: x[:1], db.split('-')))) for ngpu in ngpus for db in databackends ]
        else:
            columns = [ "{}".format(format_int(ngpu)) for ngpu in ngpus ]
        header = " {} |".format(m) + " |".join(columns) + " |"

        print(header)
        print("-"*len(header))

        for bspgpu in bspgpus:
            line = [format_int(bspgpu)]

            for ngpu in ngpus:
                for db in databackends:
                    line.append(format_float(table[m][ngpu][bspgpu][db]))

            print(" " * (len(m) - 7) + " |".join(line) + " |")

    return table


def load_baseline_file(path):
    with open(path, 'r') as f:
        baseline = json.load(f)
        return baseline['metrics']

    return None

def check(results, baseline, ngpus, bs, databackends, metrics):
    allright = True
    for m in metrics:
        for n in ngpus:
            for b in bs:
                for db in databackends:
                    result = results[m][n][b][db]
                    reference = baseline[str(n)][str(b)][m]
                    if result < PERF_THR * reference:
                        allright = False
                        print("Metric: {} NGPUs: {} BS: {} Result ( {} ) is more than {} times slower than reference ( {} )".format(
                            m, n, b, result, PERF_THR, reference))

    return allright

if args.mode == 'training':
    metrics = ['train.total_ips']
else:
    metrics = ['val.total_ips', 'val.compute_ips']

table = benchmark(args.mode, args.ngpus, args.bs, args.data_backends, command, args.bench_warmup, metrics, args.workspace)

json.dump(table, open(os.path.join(args.workspace, 'raport.json'), 'w'))

if not args.baseline is None:
    baseline = load_baseline_file(args.baseline)

    if check(table, baseline, args.ngpus, args.bs, args.data_backends, metrics):
        print("&&&& PASSED")
        exit(0)
    else:
        print("&&&& FAILED")
        exit(1)
