FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential
COPY dyana.py .
COPY dyana-requirements.txt .
COPY dyana-requirements-gpu.txt .

RUN pip install --no-cache-dir -r dyana-requirements.txt && \
    pip install --no-cache-dir -r dyana-requirements-gpu.txt

COPY main.py .

ARG EXTRA_REQUIREMENTS=""
RUN if [ ! -z "$EXTRA_REQUIREMENTS" ]; then pip install --no-cache-dir $EXTRA_REQUIREMENTS; fi

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]

