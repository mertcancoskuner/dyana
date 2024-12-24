FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

ARG EXTRA_REQUIREMENTS
RUN if [ -n "$EXTRA_REQUIREMENTS" ]; then pip install --no-cache-dir $EXTRA_REQUIREMENTS; fi

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]
