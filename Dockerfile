FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TEST_DIR=/data OUT_DIR=/output LOCAL_MODEL=/models/model

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
COPY requirements-gpu.txt .
RUN pip3 install --no-cache-dir -r requirements-gpu.txt

COPY datamate ./datamate
COPY app ./app
# COPY models/model /models/model

CMD ["python3", "-m", "datamate.main", "--backend", "local"]
