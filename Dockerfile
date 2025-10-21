FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/* && git lfs install

RUN python -m pip install --upgrade pip "setuptools<75" wheel
RUN pip install --no-cache-dir \
    xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    "flash_attn==2.7.4.post1" psutil packaging librosa requests \
    huggingface_hub hf_transfer runpod pydantic<3

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "handler.py"]
