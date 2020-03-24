NETWORK=resnet50v1.5
GEN_CONFIGS=${1:-0}

set -e

# Generate configs:

if [[ $GEN_CONFIGS == 1 ]]; then
    echo Generating configs
    rm -f ./qa/auto-gen/${NETWORK}/training/*/*
    rm -rf ./qa/auto-gen/${NETWORK}/training/*
    python3 ./qa/auto-gen/gen_training_scripts.py ./qa/auto-gen/${NETWORK}/configs.json ./qa/auto-gen/${NETWORK}/training/
    chmod +x ./qa/auto-gen/${NETWORK}/training/*.sh
fi

#Build the container

NVCR_IMG=nvidian/swdl/as-pytorch:${NETWORK}
GITLAB_IMG=gitlab-master.nvidia.com:5005/asulecki/pytorch-off/conv_tests:${NETWORK}

docker build . -t nvcr.io/$NVCR_IMG
docker tag nvcr.io/$NVCR_IMG $GITLAB_IMG
#
docker push nvcr.io/$NVCR_IMG
docker push $GITLAB_IMG

# for every DGX1 config
# run 3 runs on ngc and save job numbers
NRUN=3
for config in ./${NETWORK}/training/DGX1_*.sh; do
    echo Launching config $config
    mkdir -p ${config/.sh}_results/

    for nrun in `seq 1 $NRUN`; do
        run=True
        if ls ${config/.sh}_results/run${nrun}.jobid 1> /dev/null 2>&1; then
            jobid=`cat ${config/.sh}_results/run${nrun}.jobid`
            status=`python ./qa/auto-gen/get_status_cmd.py <(ngc batch get $jobid)`
            echo $jobid $status
            if [ ${status} = "FAILED" ]; then
                run=True
            else
                run=False
            fi
        fi
        if [ ${run} = "True" ]; then
            TIMESTAMP=`date +'%y%m%d%H%M%S%N'`

            ngc batch run \
                --name "$(basename $config)" \
                --image $NVCR_IMG --ace nv-us-west-2 --instance ngcv8 \
                --datasetid 8181:/data/imagenet --result /data/workspace \
                --commandline "/workspace/rn50/${config} /data/workspace" \
                | tee ./tmp-$TIMESTAMP

            jobid=`python3 ./qa/auto-gen/get_id_cmd.py ./tmp-$TIMESTAMP`

            echo $jobid > ${config/.sh}_results/run${nrun}.jobid
        fi
    done
done

FAILED_JOBS=0

for config in ./${NETWORK}/training/DGX1_*.sh; do
    echo Launching config $config
    mkdir -p ${config/.sh}_results/

    for nrun in `seq 1 $NRUN`; do
        if ls ${config/.sh}_results/run${nrun}.jobid 1> /dev/null 2>&1; then
            jobid=`cat ${config/.sh}_results/run${nrun}.jobid`
            status=`python ./qa/auto-gen/get_status_cmd.py <(ngc batch get $jobid)`
            echo $jobid $status
            if [ ${status} = "FAILED" ]; then
                FAILED_JOBS=$((${FAILED_JOBS}+1))
            fi
            if [ ${status} = "FINISHED_SUCCESS" ]; then
                ngc result download -d ${config/.sh}_results -f raport.json $jobid

                mv ${config/.sh}_results/${jobid}/raport.json ${config/.sh}_results/run${nrun}_log.json

                rmdir ${config/.sh}_results/${jobid}

                python ./qa/auto-gen/get_top1.py ${config/.sh}_results/run${nrun}_log.json
            fi
        fi
    done
done

echo $FAILED_JOBS failed, rerun



