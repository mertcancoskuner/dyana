FROM python:3.10-alpine

# install only essential dependencies
RUN apk update && apk add --no-cache \
    chromium \
    chromium-chromedriver

ENV PYTHONUNBUFFERED=1
ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROME_PATH=/usr/lib/chromium/
ENV DISPLAY=:99
ENV PATH="/usr/lib/chromium/:${PATH}"

# Install Python dependencies
RUN pip install --no-cache-dir selenium

WORKDIR /app
COPY dyana.py .
COPY main.py .

ENTRYPOINT ["python3", "-W", "ignore", "main.py"]