This repository contains the scripts and source code for running the MLPerf v0.6 training benchmarks written by NVIDIA in RHEL7.6 and UBI8 containers. The Dockerfiles for creating the containers are in the top level directory. 

The 4 directories, gnmt, maskrcnn, ssd, and transformer contain the source code and scripts for running each of these benchmarks. The resnet directory is for a benchmark using the ImageNet dataset on mxnet, which we haven't yet run. These 4 benchmarks are implemented in PyTorch, and the necessary PyTorch source code to be built and installed in the containers is in the ./pytorch directory. 

For more detailed information about the benchmarks, read the original documentation at [ssd](https://github.com/mlperf/training_results_v0.6/tree/master/NVIDIA/benchmarks/ssd/implementations/pytorch), [maskrcnn](https://github.com/mlperf/training_results_v0.6/tree/master/NVIDIA/benchmarks/maskrcnn/implementations/pytorch), [gnmt](https://github.com/mlperf/training_results_v0.6/tree/master/NVIDIA/benchmarks/gnmt/implementations/pytorch), [transformer](https://github.com/mlperf/training_results_v0.6/tree/master/NVIDIA/benchmarks/transformer/implementations/pytorch).

# Per-benchmark instructions
## Prerequisites
The host system needs to be configured so that containers run with podman can access GPUs.
 - Install `podman` on the host system
 - Install the `nvidia-container-toolkit`, so that the nvidia container runtime hook can be used.

## SSD

### Build the container image:
This may take ~1 hour depending on download speeds of packages.

```podman build -f ssd_dockerfile_ubi8 -t mlperf_v06_ssd_ubi8```

### Download the COCO 2017 dataset:
The data download script is in the ssd directory. Further documentation can be found in the [original results](https://github.com/mlperf/training_results_v0.6/tree/master/NVIDIA/benchmarks/ssd/implementations/pytorch).
```bash download_dataset.sh```

Put the data in whatever directory you prefer. When you run the benchmark, you will specify the directory in `DATADIR` so that the run script can mount it in a volume for the container to access.

The directory structure of `DATADIR` should be:
```
<DATADIR>
├── coco2017/
│   ├── annotations/
│   ├── models/
│   ├── train2017/
│   └── val2017/
...
```

### Run the benchmark
From the `ssd` directory, the benchmark can be started with
```CONT=mlperf_v06_ssd_ubi8 DATADIR=<coco2017> LOGDIR=/data/mlperf/logs DGXSYSTEM=DGX1 NEXP=1 PULL=0 ./podman_run.sub```

On a system other than an NVIDIA DGX1 or DGX2, you will likely need to create a custom `config_*.sh` file to specify the number of GPUs, and to configure other parameters to optimize performance.

## Mask R-CNN

### Build the container image:
This may take ~1 hour depending on download speeds of packages.

```podman build -f maskrcnn_dockerfile_ubi8 -t mlperf_v06_maskrcnn_ubi8```

### Download the COCO 2017 dataset:
The data is the same as that which is used for the ssd benchmark.   See the above download instructions.

### Download the ResNet-50 weights
The Mask R-CNN uses the trained ResNet-50 model as a backbone. Run the `download_weights.sh` script in the `maskrcnn/` directory. Place the resulting file `R-50.pkl` in the directory `coco2017/models` directory, where `coco2017` contains the unzipped `train2017` and `val2017` directories.

### Run the benchmark
From the `maskrcnn` directory, the benchmark can be started with
```CONT=mlperf_v06_maskrcnn_ubi8 DATADIR=<coco2017> LOGDIR=/data/mlperf/logs DGXSYSTEM=DGX1 NEXP=1 PULL=0 ./podman_run.sub```

On a system other than an NVIDIA DGX1 or DGX2, you will likely need to create a custom `config_*.sh` file to specify the number of GPUs, and to configure other parameters to optimize performance.

## GNMT

### Build the container image:
This may take ~1 hour depending on download speeds of packages.

```podman build -f gnmt_dockerfile_ubi8 -t mlperf_v06_gnmt_ubi8```

### Download the data set:
Follow the instructions in the [original documentation](https://github.com/mlperf/training_results_v0.6/tree/master/NVIDIA/benchmarks/gnmt/implementations/pytorch#2-directions).

### Run the benchmark
From the `gnmt` directory, the benchmark can be started with
```CONT=mlperf_v06_gnmt_ubi8 DATADIR=<gnmt_data> LOGDIR=/data/mlperf/logs DGXSYSTEM=DGX1 NEXP=1 PULL=0 ./podman_run.sub```

On a system other than an NVIDIA DGX1 or DGX2, you will likely need to create a custom `config_*.sh` file to specify the number of GPUs, and to configure other parameters to optimize performance.

## Transformer

### Build the container image:
This may take ~1 hour depending on download speeds of packages.

```podman build -f transformer_dockerfile_ubi8 -t mlperf_v06_transformer_ubi8```

### Download the data set:
Unfortunately the scripts in the submission files for the transformer benchmark do not work. To run the benchmark we had to get the data directly from the creators. I am still working on finding the exact source for the correct data.

### Run the benchmark
From the `transformer` directory, the benchmark can be started with
```CONT=mlperf_v06_gnmt_ubi8 DATADIR=<transformer_data> LOGDIR=/data/mlperf/logs DGXSYSTEM=DGX1 NEXP=1 PULL=0 ./podman_run.sub```

On a system other than an NVIDIA DGX1 or DGX2, you will likely need to create a custom `config_*.sh` file to specify the number of GPUs, and to configure other parameters to optimize performance.
