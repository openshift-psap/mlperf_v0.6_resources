NETWORK=resnet50v1.5
GEN_CONFIGS=${1:-0}

set -e

# Generate configs:

if [[ $GEN_CONFIGS == 1 ]]; then
    echo Generating benchmarks
    rm -f ./qa/auto-gen/${NETWORK}/benchmark/*/*
    rm -rf ./qa/auto-gen/${NETWORK}/benchmark/*
    python3 ./qa/auto-gen/gen_training_scripts.py ./qa/auto-gen/${NETWORK}/bench-configs.json ./qa/auto-gen/${NETWORK}/benchmark/
    chmod +x ./qa/auto-gen/${NETWORK}/benchmark/*.sh
fi

#Build the container

NVCR_IMG=nvidian/swdl/as-pytorch:${NETWORK}
GITLAB_IMG=gitlab-master.nvidia.com:5005/asulecki/pytorch-off/conv_tests:${NETWORK}

docker build . -t nvcr.io/$NVCR_IMG
docker tag nvcr.io/$NVCR_IMG $GITLAB_IMG
#
docker push nvcr.io/$NVCR_IMG
docker push $GITLAB_IMG

for benchmark in ./qa/auto-gen/${NETWORK}/benchmark/*.sh; do
    echo Processing benchmark $benchmark

    case "$(basename $benchmark)" in
        *DGX1* ) partition=dgx1v16g;;
        *DGX2* ) partition=xpl;;
    esac

    benchdir=${benchmark/.sh}_result/
    mkdir -p ${benchdir}
    chmod 777 ${benchdir}

    workdir=$(pwd)

    if ls $benchdir/raport.json 1> /dev/null 2>&1; then
        echo BENCHMARK ALREADY FINISHED
    else
        if ls $benchdir/run 1> /dev/null 2>&1; then
            jobid=`cat $benchdir/run`
            echo JOB SUBMMITED $jobid
        else
            echo LAUNCHING
            touch $benchdir/run
            ssh -q asulecki@dbcluster <<EOSSH
cd $workdir

sbatch -o ./${benchdir}/%j-slurmlog.log -p $partition <<EOSBATCH | awk '{ print $4 }' | tee $workdir/$benchdir/run
#!/bin/bash

cd $workdir
docker pull ${GITLAB_IMG}
nvidia-docker run --rm --ipc=host \
    -v /raid/dldata/imagenet/train-val-recordio-passthrough/:/data/imagenet-recio \
    -v /raid/dldata/imagenet/train-jpeg:/data/imagenet/train \
    -v /raid/dldata/imagenet/val-jpeg:/data/imagenet/val \
    -v $workdir/$benchdir:/data/workspace \
    ${GITLAB_IMG} /bin/bash -c "$benchmark /data/workspace"
EOSBATCH

EOSSH
        fi
    fi
done



