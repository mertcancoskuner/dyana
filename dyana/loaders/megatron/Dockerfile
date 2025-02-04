FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_USE_CUDA_DSA=0
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_ASYNC_ERROR_HANDLING=1
ENV OMP_NUM_THREADS=1
ENV NVTE_FRAMEWORK=pytorch
ENV MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_ALLOW_TF32=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV PYTORCH_JIT=0
ENV TORCH_INDUCTOR_DISABLE_CUDA_GRAPH=1
ENV TORCH_INDUCTOR_USE_PYTHON_BINDING=0
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV NCCL_IB_DISABLE=1
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1
ENV TORCH_SHOW_CPP_STACKTRACES=0
ENV PYTHONWARNINGS=ignore

# Only verify PyTorch during build
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Create working directory
RUN mkdir -p /app/workspace

# Copy files in correct order
COPY requirements.txt /app/workspace/
COPY *.py /app/workspace/
COPY dyana-requirements*.txt /app/workspace/

WORKDIR /app/workspace

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Megatron-LM
RUN git clone --depth 1 --branch dmc https://github.com/NVIDIA/Megatron-LM.git /app/Megatron-LM && \
    cd /app/Megatron-LM && \
    pip install -e .

ENV PYTHONPATH=/app/Megatron-LM:$PYTHONPATH

# Create directories for IPC
RUN mkdir -p /dev/shm && \
    mkdir -p /tmp/pytorch_extensions && \
    chmod -R 777 /dev/shm /tmp/pytorch_extensions

# Create simpler entrypoint script with proper environment
RUN printf '#!/bin/bash\n\
    # Clear any stale semaphores\nrm -rf /dev/shm/* 2>/dev/null\n\
    export PYTHONPATH=/app/workspace:/app/Megatron-LM:$PYTHONPATH\n\
    exec python3 -W ignore main.py "$@"\n' > /app/workspace/entrypoint.sh && \
    chmod +x /app/workspace/entrypoint.sh

# Verify files exist and have correct permissions
RUN ls -la /app/workspace && \
    ls -la /app/workspace/entrypoint.sh && \
    test -x /app/workspace/entrypoint.sh

# Set proper ownership and permissions
RUN chown -R root:root /app && \
    chmod -R 755 /app && \
    chmod +x /app/workspace/entrypoint.sh

# Use bash as entrypoint shell
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "exec /app/workspace/entrypoint.sh \"$@\""]