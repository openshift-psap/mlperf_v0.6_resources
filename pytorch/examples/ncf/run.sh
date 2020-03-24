#!/bin/bash
set -e

# runs benchmark and reports time to convergence

BASEDIR=$(dirname -- "$0")

BATCH=${BATCH:-1048576}
LR=${LR:-0.0045}
BETA1=${BETA1:-0.25}
BETA2=${BETA2:-0.5}
WORLD_SIZE=${WORLD_SIZE:-8}
LOSS_SCALE=${LOSS_SCALE:-8192}
EPS=${EPS:-1e-8}
NUM_RUNS=${NUM_RUNS:-1}
THRESHOLD=${THRESHOLD:-1}
VALID_NEGATIVE=${VALID_NEGATIVE:-100}

if [ -z "$ADDITIONAL_FLAGS" ]; then
    ADDITIONAL_FLAGS=' --fp16 '
fi

if [ $WORLD_SIZE -eq 8 ]
then
    valid_batch_size=524288
elif [ $WORLD_SIZE -eq 16 ]
then
    valid_batch_size=524288
else
    valid_batch_size=262144
fi

DATADIR='/mlperf/recommendation/pytorch/ml-20m/cache'
mkdir -p $DATADIR
rm -f log

if [ ! -f ml-20m/ml-20m.zip ]; then
    echo 'Dataset not found, downloading...'
    ./download_dataset.sh
fi

if [ ! -f ml-20m/ratings.csv ]; then
    unzip -u ml-20m/ml-20m.zip
fi

if [ ! -f ${DATADIR}/train_ratings.pt ]; then
    echo "preprocessing ratings.csv and save to disk"
    t0=$(date +%s)
    python convert.py --path ml-20m/ratings.csv --output ${DATADIR} >> log
    t1=$(date +%s)
    delta=$(( $t1 - $t0 ))
    echo "Finish preprocessing in $delta seconds"
else
    echo 'Using cached preprocessed data'
fi

for i in $(seq 1 $NUM_RUNS)
do
    t0=$(date +%s)
    python -m torch.distributed.launch --nproc_per_node=$WORLD_SIZE $BASEDIR/ncf.py $DATADIR -l ${LR} -b ${BATCH} -b1 ${BETA1} \
           -b2 ${BETA2} --eps ${EPS} --valid-batch-size $valid_batch_size --loss-scale ${LOSS_SCALE} --layers 256 256 128 64 -f 64 \
           --seed ${i} --threshold ${THRESHOLD} --valid-negative ${VALID_NEGATIVE} ${ADDITIONAL_FLAGS} $@ | tee nv.log
    t1=$(date +%s)
    delta=$(( $t1 - $t0 ))
    echo "Time: $delta seconds"
    python -m logger.analyzer nv.log > nvlog.json
done
