FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# python deps
RUN pip install --upgrade pip && \
    pip install runpod requests fastapi uvicorn "huggingface_hub<1.0" && \
    pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 || true

# copy your project (we need generate_infinitetalk.py + repo files)
COPY . /app

# project requirements (best-effort)
RUN pip install -r /app/requirements.txt || true
RUN pip install "flash-attn==2.5.8" --no-build-isolation || true

# start serverless worker (NOT uvicorn)
CMD ["python", "-u", "/app/handler.py"]
