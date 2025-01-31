FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Megatron-LM dmc branch and install
RUN git clone -b dmc https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    pip install -e .

# Copy loader files
COPY . .

# Install requirements
RUN pip install --no-cache-dir -r dyana-requirements-gpu.txt
RUN pip install --no-cache-dir -r requirements.txt

# Environment setup
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV PYTHONUNBUFFERED=1
ENV MEGATRON_DEBUG=1

ENTRYPOINT ["python3", "-u", "main.py"]