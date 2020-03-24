# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:18.12-py3-devel

RUN apt-get update && \
    apt-get install -y unzip

# assume build from mlperf top level folder
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . /mlperf/recommendation
WORKDIR /mlperf/recommendation
