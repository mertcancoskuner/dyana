FROM ollama/ollama

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gnupg build-essential curl
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main" > /etc/apt/sources.list.d/deadsnakes.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys f23c5a6cf475977595c89f51ba6932366a755776
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=America/New_York apt-get install -y --no-install-recommends python3.12

RUN curl https://bootstrap.pypa.io/get-pip.py -o pip.py
RUN python3.12 pip.py

COPY dyana.py .
COPY dyana-requirements-gpu.txt .
RUN pip install --no-cache-dir -r dyana-requirements-gpu.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dyana.py main.py ./

ENTRYPOINT ["python3.12", "-W", "ignore", "main.py"]