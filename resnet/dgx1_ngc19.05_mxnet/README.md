## Steps to launch training

### NVIDIA DGX-1 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
single node submission are in the `config_DGX1.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

```
cd ../implementations/mxnet
docker build --pull -t mlperf-nvidia:image_classification .
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1 ./run.sub
```
