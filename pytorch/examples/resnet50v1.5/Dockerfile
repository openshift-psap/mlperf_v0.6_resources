FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.04-py3-stage

RUN git clone https://github.com/NVIDIA/apex \
        && cd apex \
        && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

ADD . /workspace/rn50
WORKDIR /workspace/rn50
