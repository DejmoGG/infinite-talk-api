FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg curl python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

# PyTorch + CUDA 12.1 wheels (no source build)
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Clone InfiniteTalk
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /app/InfiniteTalk

# Project deps
RUN pip install --no-cache-dir -r /app/InfiniteTalk/requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn requests huggingface_hub "pydantic<3" \
                      misaki[en] ninja psutil packaging wheel && \
    pip install --no-cache-dir xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 && \
    # flash-attn is optional; don't kill the build if no wheel
    pip install --no-cache-dir flash-attn==2.5.8 --no-build-isolation || true

COPY server.py /app/server.py
RUN mkdir -p /app/weights

EXPOSE 8000
CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000"]
