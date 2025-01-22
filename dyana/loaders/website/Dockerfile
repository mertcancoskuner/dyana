FROM python:3.10-alpine

# install chromedriver
RUN apk update
RUN apk add chromium chromium-chromedriver

ENV PYTHONUNBUFFERED 1
RUN pip install --upgrade pip
RUN pip install selenium

WORKDIR /app
COPY dyana.py .
COPY main.py .

ENV DISPLAY=:99

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]