FROM nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget ffmpeg python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN python3 -m pip install --upgrade pip

# 3. Clone InfiniteTalk source
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git
WORKDIR /app/InfiniteTalk

# 4. Install required Python packages
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install misaki[en] ninja psutil packaging wheel && \
    pip install flash_attn==2.7.4.post1 && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn requests huggingface_hub "pydantic<3"

# 5. Go back to main app folder and copy API server
WORKDIR /app
COPY server.py /app/server.py

# 6. Create folder for model weights (they will be downloaded on first run)
RUN mkdir -p /app/weights

# 7. Expose the FastAPI port
EXPOSE 8000

# 8. Start the API when container runs
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
