FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create required directories for multiprocessing
RUN mkdir -p /dev/shm && \
    mkdir -p /tmp/pytorch_extensions && \
    mkdir -p /run/shm && \
    chmod -R 777 /dev/shm /tmp/pytorch_extensions /run/shm

# Create ALL required directories for IPC and shared memory
RUN mkdir -p /dev/shm && \
    mkdir -p /run/shm && \
    mkdir -p /tmp/pytorch_extensions && \
    mkdir -p /tmp/.pytorch_jit_cache && \
    mkdir -p /tmp/transformers && \
    chmod -R 777 /dev/shm /run/shm /tmp/pytorch_extensions /tmp/.pytorch_jit_cache /tmp/transformers

# Only verify PyTorch version during build (not CUDA)
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

# Create entrypoint script
RUN printf '#!/bin/bash\n\
    python3 -c "import torch; assert torch.cuda.is_available(), \"CUDA is not available\"; device=torch.cuda.get_device_name(); print(f\"CUDA OK: {device}\")" && \
    export PYTHONPATH=/app/workspace:/app/Megatron-LM:$PYTHONPATH && \
    exec python3 -W ignore main.py "$@"' > /app/workspace/entrypoint.sh && \
    chmod +x /app/workspace/entrypoint.sh

# Verify files exist and perms
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