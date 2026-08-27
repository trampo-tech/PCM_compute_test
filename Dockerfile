# syntax=docker/dockerfile:1
ARG BASE_IMAGE=pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
FROM ${BASE_IMAGE}

# --- Build-time knobs -------------------------------------------------------
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    PIP_CONSTRAINT=/tmp/pip-constraints.txt \
    PYTHONDONTWRITEBYTECODE=1 \
    CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=5G \
    PYTHONPATH="/workspace/PointCloudMamba" \
    PATH=/usr/lib/ccache:${PATH}

# Pin numpy, setuptools, and transformers to avoid breaking PyTorch 2.1 / Mamba
RUN printf 'numpy<2\nsetuptools<70\ntransformers==4.36.2\n' > /tmp/pip-constraints.txt

# --- System build dependencies ---------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build ccache \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc.real \
    && printf '#!/bin/sh\nexec ccache /usr/local/cuda/bin/nvcc.real "$@"\n' > /usr/local/cuda/bin/nvcc \
    && chmod +x /usr/local/cuda/bin/nvcc

WORKDIR /workspace/PointCloudMamba

# --- 1) Base dependencies & runtime requirements ---------------------------
# Filter out any upstream mamba-ssm / causal-conv1d lines from requirements.txt
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install "setuptools<70" wheel ninja packaging "numpy<2" "transformers==4.36.2" einops \
    && pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html \
    && sed -i '/mamba/d; /causal_conv1d/d; /causal-conv1d/d' ./requirements.txt \
    && pip install -r requirements.txt

# --- 2) OpenPoints CUDA extensions -----------------------------------------
COPY openpoints/cpp/ ./openpoints/cpp/
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    cd openpoints/cpp/pointnet2_batch && pip install --no-build-isolation . \
    && cd ../subsampling && python setup.py build_ext --inplace \
    && cd ../pointops && pip install --no-build-isolation . \
    && if [ -d "../chamfer_dist" ]; then cd ../chamfer_dist && pip install --no-build-isolation .; fi \
    && if [ -d "../emd" ]; then cd ../emd && pip install --no-build-isolation .; fi

# --- 3) PointCloudMamba custom Causal Conv1D & Bi-Mamba packages -----------
COPY openpoints/models/PCM/ ./openpoints/models/PCM/
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    pip uninstall -y mamba-ssm causal-conv1d || true \
    && cd openpoints/models/PCM/causal-conv1d \
    && pip install --no-build-isolation --no-deps -e . \
    && cd ../mamba \
    && pip install --no-build-isolation --no-deps -e .

# --- 4) Copy entire repo ----------------------------------------------------
COPY . .
RUN cd openpoints/cpp/subsampling && python setup.py build_ext --inplace

# --- 5) Build-time Smoke Test -----------------------------------------------
RUN python -c "import torch, torch_scatter, pointnet2_batch_cuda, pointops_cuda, transformers; \
from mamba_ssm.modules.mamba_simple import Mamba; \
import inspect; \
print('Mamba init args:', inspect.signature(Mamba.__init__)); \
m = Mamba(d_model=64, bimamba_type='v2'); \
print('CUDA Available:', torch.cuda.is_available()); \
print('Bi-Mamba successfully loaded from:', Mamba.__module__)"

CMD ["/bin/bash"]
