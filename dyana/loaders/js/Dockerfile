FROM python:3.12-slim

RUN apt-get update && apt-get install -y nodejs npm

WORKDIR /app

RUN apt-get install -y build-essential
COPY dyana.py .
COPY dyana-requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r dyana-requirements.txt

COPY main.py .

ARG EXTRA_REQUIREMENTS
RUN if [ -n "$EXTRA_REQUIREMENTS" ]; then npm install $EXTRA_REQUIREMENTS; fi

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]