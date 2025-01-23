FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential
COPY dyana.py .
COPY dyana-requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r dyana-requirements.txt

COPY requirements.txt .
COPY main.py .

RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

ARG EXTRA_REQUIREMENTS
RUN if [ -n "$EXTRA_REQUIREMENTS" ]; then pip install --no-cache-dir --root-user-action=ignore $EXTRA_REQUIREMENTS; fi

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]
