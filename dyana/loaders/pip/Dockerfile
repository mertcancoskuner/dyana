FROM python:3.12-slim

WORKDIR /app

COPY dyana.py .
COPY main.py .

ARG EXTRA_DEPENDENCIES
RUN if [ -n "$EXTRA_DEPENDENCIES" ]; then apt-get update && apt-get install -yq $EXTRA_DEPENDENCIES; fi

RUN pip3 install cython

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]