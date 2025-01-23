FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential
COPY dyana.py .
COPY dyana-requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r dyana-requirements.txt

COPY main.py .

ARG EXTRA_DEPENDENCIES
RUN if [ -n "$EXTRA_DEPENDENCIES" ]; then apt-get update && apt-get install -yq $EXTRA_DEPENDENCIES; fi

RUN pip3 install cython
RUN pip3 install --upgrade setuptools

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]