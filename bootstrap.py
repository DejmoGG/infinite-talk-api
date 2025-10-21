import os, pathlib, subprocess

VOL = pathlib.Path(os.getenv("RUNPOD_VOLUME", "/runpod-volume"))
WEIGHTS = VOL / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
env.setdefault("HF_HOME", str(VOL / "hf"))
env.setdefault("HF_HUB_CACHE", str(VOL / "hf_cache"))

def sh(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.check_call(args, env=env)

def dl(model, dest, *extra):
    dst = WEIGHTS / dest
    if dst.exists():
        print(f"✓ {model} cached at {dst}", flush=True)
        return
    sh("huggingface-cli","download",model,"--local-dir",str(dst), *extra)

dl("Wan-AI/Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-480P")
dl("TencentGameMate/chinese-wav2vec2-base", "chinese-wav2vec2-base")
dl("TencentGameMate/chinese-wav2vec2-base", "chinese-wav2vec2-base",
   "model.safetensors","--revision","refs/pr/1")
dl("MeiGen-AI/InfiniteTalk", "InfiniteTalk")

print("Bootstrap complete.", flush=True)
