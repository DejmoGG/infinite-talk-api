FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/* && git lfs install

RUN python -m pip install --upgrade pip "setuptools<75" wheel

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "/app/handler.py"]
