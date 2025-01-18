FROM python:3.12-slim

RUN apt-get update && apt-get install -y nodejs npm

ARG EXTRA_DEPENDENCIES
RUN if [ -n "$EXTRA_DEPENDENCIES" ]; then apt-get install -yq $EXTRA_DEPENDENCIES; fi

WORKDIR /app

COPY dyana.py .
COPY main.py .

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]