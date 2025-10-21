FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1) + xformers
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Core deps (README-aligned) + tools we use
RUN pip install --no-cache-dir \
    misaki[en] ninja psutil packaging wheel \
    "flash_attn==2.7.4.post1" \
    fastapi uvicorn requests "pydantic<3" librosa \
    huggingface_hub hf_transfer

# Clone InfiniteTalk
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /app/InfiniteTalk

# Install project requirements
RUN pip install --no-cache-dir -r /app/InfiniteTalk/requirements.txt || true

# Create weights dir
RUN mkdir -p /app/weights

# ----- Download required weights (build-time) -----
# (If you have a HF token for faster/guaranteed pulls, pass it as build arg.)
ARG HF_TOKEN=""
ENV HF_TOKEN=$HF_TOKEN

# Wan 14B base model
RUN bash -lc ' \
    if [ -n "$HF_TOKEN" ]; then export HF_TOKEN_ENV="--token $HF_TOKEN"; else HF_TOKEN_ENV=""; fi; \
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /app/weights/Wan2.1-I2V-14B-480P $HF_TOKEN_ENV \
'

# Chinese wav2vec base + PR weights
RUN bash -lc ' \
    if [ -n "$HF_TOKEN" ]; then export HF_TOKEN_ENV="--token $HF_TOKEN"; else HF_TOKEN_ENV=""; fi; \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir /app/weights/chinese-wav2vec2-base $HF_TOKEN_ENV && \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir /app/weights/chinese-wav2vec2-base $HF_TOKEN_ENV \
'

# InfiniteTalk conditioning weights
RUN bash -lc ' \
    if [ -n "$HF_TOKEN" ]; then export HF_TOKEN_ENV="--token $HF_TOKEN"; else HF_TOKEN_ENV=""; fi; \
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir /app/weights/InfiniteTalk $HF_TOKEN_ENV \
'

# Your serverless handler (RunPod) OR server.py for API (not used in serverless)
COPY handler.py /app/handler.py

# Expose (not actually used by serverless, but harmless)
EXPOSE 8000

# Serverless: run handler
CMD ["python","-u","/app/handler.py"]
