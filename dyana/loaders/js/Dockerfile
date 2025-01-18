FROM python:3.12-slim

RUN apt-get update && apt-get install -y nodejs npm

WORKDIR /app

COPY dyana.py .
COPY main.py .

ARG EXTRA_REQUIREMENTS
RUN if [ -n "$EXTRA_REQUIREMENTS" ]; then npm install $EXTRA_REQUIREMENTS; fi

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]