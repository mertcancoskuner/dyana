FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]
