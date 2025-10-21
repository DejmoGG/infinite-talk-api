# ===== Base image =====
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# ===== System deps =====
# - ffmpeg (media)
# - libsndfile1 (needed by librosa/soundfile)
# - git, curl, python pip, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg libsndfile1 python3-pip build-essential \
 && rm -rf /var/lib/apt/lists/*

# ===== Python core =====
RUN python3 -m pip install --upgrade pip

# PyTorch/cu121
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# xformers (cu121 wheel)
RUN pip install --no-cache-dir \
    xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Base libs + runtime (fastapi/uvicorn not strictly needed for Queue, harmless)
RUN pip install --no-cache-dir \
    misaki[en] ninja psutil packaging wheel \
    "flash_attn==2.7.4.post1" \
    requests "pydantic<3" librosa \
    huggingface_hub hf_transfer \
    runpod

# ===== InfiniteTalk source =====
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /app/InfiniteTalk

# Some repos mark optional extras; don't fail the build if something is optional
RUN pip install --no-cache-dir -r /app/InfiniteTalk/requirements.txt || true

# ===== Weights layout =====
RUN mkdir -p /app/weights

# Wan 14B base model
RUN huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir /app/weights/Wan2.1-I2V-14B-480P

# Chinese wav2vec base + PR model.safetensors
RUN huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    --local-dir /app/weights/chinese-wav2vec2-base && \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors \
    --revision refs/pr/1 --local-dir /app/weights/chinese-wav2vec2-base

# InfiniteTalk conditioning weights
RUN huggingface-cli download MeiGen-AI/InfiniteTalk \
    --local-dir /app/weights/InfiniteTalk

# ===== Serverless Handler =====
COPY handler.py /app/handler.py

# (Not used by Queue, but harmless if you later expose HTTP)
EXPOSE 8000

# ===== Serverless entrypoint =====
CMD ["python","-u","/app/handler.py"]
