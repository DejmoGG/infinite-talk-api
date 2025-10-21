# handler.py
import os, uuid, json, pathlib, subprocess, requests
import runpod

WEIGHTS_ROOT = pathlib.Path("/app/weights")

CKPT_DIR = WEIGHTS_ROOT / "Wan2.1-I2V-14B-480P"
WAV2VEC_DIR = WEIGHTS_ROOT / "chinese-wav2vec2-base"
INF_TALK_DIR = WEIGHTS_ROOT / "InfiniteTalk" / "single" / "infinitetalk.safetensors"

GEN_SCRIPT = "/app/InfiniteTalk/generate_infinitetalk.py"

def _download(url: str, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                if chunk:
                    f.write(chunk)

def _run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])
    return proc.stdout

def handler(event):
    """
    Request format (Queue):
    {
      "input": {
        "image_url": "https://...png|jpg",
        "audio_url": "https://...mp3|wav",
        "quality": "720p"|"480p"  (use 720p or 480p; we will optionally upscale later)
      }
    }
    """
    inp = event.get("input") or {}
    image_url = inp.get("image_url")
    audio_url = inp.get("audio_url")
    quality   = (inp.get("quality") or "720p").lower()

    if not image_url or not audio_url:
        return {"status": "error", "error": "image_url and audio_url are required"}

    job = pathlib.Path("/tmp") / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=True)

    image_path = job / "input.png"
    audio_src  = job / "audio_src"
    audio_wav  = job / "input.wav"
    request_js = job / "request.json"

    # 1) Download inputs
    try:
        _download(image_url, image_path)
        _download(audio_url, audio_src)
    except Exception as e:
        return {"status": "error", "error": f"download failed: {e}"}

    # 2) Normalize audio (16k, mono, pcm_s16le)
    try:
        _run([
            "ffmpeg", "-y", "-i", str(audio_src),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(audio_wav)
        ])
    except Exception as e:
        return {"status": "error", "error": f"ffmpeg convert failed: {e}"}

    # 3) Build the input JSON expected by the script (image/audio path references)
    with open(request_js, "w") as f:
        json.dump({
            "image": str(image_path),
            "audio": str(audio_wav)
        }, f)

    size_flag = "infinitetalk-720" if quality == "720p" else "infinitetalk-480"
    save_prefix = "infinitetalk_res"

    # 4) Generate (per README flags)
    cmd = [
        "python", GEN_SCRIPT,
        "--ckpt_dir", str(CKPT_DIR),
        "--wav2vec_dir", str(WAV2VEC_DIR),
        "--infinitetalk_dir", str(INF_TALK_DIR),
        "--input_json", str(request_js),
        "--size", size_flag,
        "--sample_steps", "40",
        "--mode", "streaming",
        "--motion_frame", "9",
        "--save_file", save_prefix
    ]

    try:
        gen_log = _run(cmd, cwd=str(job))
    except Exception as e:
        return {"status": "error", "error": f"InfiniteTalk failed: {e}"}

    # 5) Find output
    mp4 = None
    for p in job.glob(f"{save_prefix}*.mp4"):
        mp4 = p
        break
    if not mp4:
        # last-chance scan
        cands = list(job.glob("*.mp4"))
        mp4 = cands[0] if cands else None
    if not mp4:
        return {"status": "error", "error": "No MP4 produced"}

    # 6) Upload to file.io
    try:
        up = subprocess.check_output(["curl", "-sF", f"file=@{mp4}", "https://file.io"]).decode("utf-8")
        link = json.loads(up)["link"]
    except Exception as e:
        return {"status": "error", "error": f"upload failed: {e}"}

    return {"status": "done", "video_url": link}

runpod.serverless.start({"handler": handler})
