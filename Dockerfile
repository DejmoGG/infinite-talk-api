# Use the official PyTorch image that already matches CUDA 12.1 + cuDNN 8
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Clone InfiniteTalk
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /app/InfiniteTalk

# Python deps
# torch 2.4.1 is already in the base image; install matching extras + project reqs
RUN pip install --no-cache-dir \
    torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /app/InfiniteTalk/requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn requests huggingface_hub "pydantic<3" && \
    pip install --no-cache-dir xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 && \
    # FlashAttention prebuilt wheel; if no wheel for this arch, don't fail the build
    pip install --no-cache-dir flash-attn==2.5.8 --no-build-isolation || true

# Your API
COPY server.py /app/server.py

# Weights folder (downloaded on first run)
RUN mkdir -p /app/weights

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
