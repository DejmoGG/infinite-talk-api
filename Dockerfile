FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl ffmpeg libsndfile1 ca-certificates && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# Python deps that need specific wheels for CUDA 12.1
RUN python -m pip install --upgrade pip "setuptools<75" wheel
RUN pip install --no-cache-dir \
    xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    "flash_attn==2.7.4.post1" \
    runpod requests psutil packaging librosa huggingface_hub hf_transfer

# Pull InfiniteTalk code (we do NOT need it in your repo)
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /app/InfiniteTalk
RUN pip install --no-cache-dir -r /app/InfiniteTalk/requirements.txt

# Your bootstrap + handler
COPY bootstrap.py /app/bootstrap.py
COPY handler.py   /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
