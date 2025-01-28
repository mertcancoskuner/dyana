FROM ollama/ollama

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential python3-pip python3
COPY dyana.py .
COPY dyana-requirements-gpu.txt .
RUN pip install --no-cache-dir -r dyana-requirements-gpu.txt

COPY requirements.txt .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]