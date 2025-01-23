FROM python:3.12-slim

RUN apt-get update && apt-get install -y nodejs npm

ARG EXTRA_DEPENDENCIES
RUN if [ -n "$EXTRA_DEPENDENCIES" ]; then apt-get install -yq $EXTRA_DEPENDENCIES; fi

WORKDIR /app

RUN apt-get install -y build-essential
COPY dyana.py .
COPY dyana-requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r dyana-requirements.txt

COPY main.py .

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]