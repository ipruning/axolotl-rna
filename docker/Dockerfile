ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG MAX_JOBS=4

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION AS base-builder

ENV PATH="/root/miniconda3/bin:${PATH}"

ARG PYTHON_VERSION="3.10"
ARG PYTORCH_VERSION="2.1.2"
ARG CUDA="118"
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"

ENV PYTHON_VERSION=$PYTHON_VERSION
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

RUN apt-get update \
    && apt-get install -y wget git build-essential ninja-build git-lfs libaio-dev pkg-config && rm -rf /var/lib/apt/lists/* \
    && wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

ENV PATH="/root/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}"

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip && pip3 install packaging && \
    python3 -m pip install --no-cache-dir -U torch==${PYTORCH_VERSION}+cu${CUDA} --extra-index-url https://download.pytorch.org/whl/cu$CUDA && \
    python3 -m pip install --no-cache-dir "causal_conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@main" && \
    python3 -m pip install --no-cache-dir "mamba_ssm @ git+https://github.com/state-spaces/mamba.git@main"

RUN git lfs install --skip-repo && \
    pip3 install awscli && \
    # The base image ships with `pydantic==1.8.2` which is not working
    pip3 install -U --no-cache-dir pydantic==1.10.10

ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ARG AXOLOTL_EXTRAS=""
ARG AXOLOTL_ARGS=""
ARG CUDA="118"
ARG PYTORCH_VERSION="2.1.2"

ENV PYTORCH_VERSION=$PYTORCH_VERSION

RUN apt-get update && \
    apt-get install -y --allow-change-held-packages vim curl nano libnccl2 libnccl-dev rsync s3fs

WORKDIR /workspace

RUN git clone --depth=1 https://github.com/axolotl-ai-cloud/axolotl.git

WORKDIR /workspace/axolotl

# If AXOLOTL_EXTRAS is set, append it in brackets
RUN if [ "$AXOLOTL_EXTRAS" != "" ] ; then \
    pip install --no-build-isolation -e .[deepspeed,flash-attn,optimizers,ray,$AXOLOTL_EXTRAS] $AXOLOTL_ARGS; \
    else \
    pip install --no-build-isolation -e .[deepspeed,flash-attn,optimizers,ray] $AXOLOTL_ARGS; \
    fi

RUN python scripts/unsloth_install.py | sh
RUN python scripts/cutcrossentropy_install.py | sh

# So we can test the Docker image
RUN pip install pytest

# fix so that git fetch/pull from remote works
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && \
    git config --get remote.origin.fetch

# helper for huggingface-login cli
RUN git config --global credential.helper store
