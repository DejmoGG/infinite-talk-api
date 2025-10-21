import os, uuid, json, pathlib, subprocess, requests, runpod

# Warm models into the mounted volume
subprocess.run(["python","/app/bootstrap.py"], check=True)

VOLUME_ROOT  = pathlib.Path(os.getenv("RUNPOD_VOLUME","/runpod-volume"))
WEIGHTS_ROOT = VOLUME_ROOT / "weights"

CKPT_DIR    = WEIGHTS_ROOT / "Wan2.1-I2V-14B-480P"
WAV2VEC_DIR = WEIGHTS_ROOT / "chinese-wav2vec2-base"
INF_TALK    = WEIGHTS_ROOT / "InfiniteTalk" / "single" / "infinitetalk.safetensors"
GEN_SCRIPT  = "/app/InfiniteTalk/generate_infinitetalk.py"

def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                if chunk: f.write(chunk)

def _run(cmd, cwd=None):
    print(">>", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout[-4000:])
    return p.stdout

def handler(event):
    """
    input: { "image_url": "...", "audio_url": "...", "quality": "720p"|"480p" }
    """
    inp = event.get("input") or {}
    image_url = inp.get("image_url")
    audio_url = inp.get("audio_url")
    quality   = (inp.get("quality") or "720p").lower()
    if not image_url or not audio_url:
        return {"status":"error","error":"image_url and audio_url are required"}

    job = pathlib.Path("/tmp") / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=True)

    img = job / "input.png"
    aud_src = job / "audio_src"
    aud_wav = job / "input.wav"
    req_js  = job / "request.json"

    try:
        _download(image_url, img)
        _download(audio_url, aud_src)
    except Exception as e:
        return {"status":"error","error":f"download failed: {e}"}

    try:
        _run(["ffmpeg","-y","-i",str(aud_src),"-ar","16000","-ac","1","-c:a","pcm_s16le",str(aud_wav)])
    except Exception as e:
        return {"status":"error","error":f"ffmpeg convert failed: {e}"}

    req_js.write_text(json.dumps({"image": str(img), "audio": str(aud_wav), "seed": 1}))
    size = "infinitetalk-720" if quality=="720p" else "infinitetalk-480"

    cmd = [
        "python", GEN_SCRIPT,
        "--ckpt_dir", str(CKPT_DIR),
        "--wav2vec_dir", str(WAV2VEC_DIR),
        "--infinitetalk_dir", str(INF_TALK),
        "--input_json", str(req_js),
        "--size", size,
        "--sample_steps", "40",
        "--mode", "streaming",
        "--motion_frame", "9",
        "--save_file", "infinitetalk_res"
    ]
    try:
        _run(cmd, cwd=str(job))
    except Exception as e:
        return {"status":"error","error":f"InfiniteTalk failed: {e}"}

    mp4 = next(iter(job.glob("infinitetalk_res*.mp4")), None) or next(iter(job.glob("*.mp4")), None)
    if not mp4:
        return {"status":"error","error":"No MP4 produced"}

    try:
        up = subprocess.check_output(["curl","-sF",f"file=@{mp4}","https://file.io"]).decode()
        link = json.loads(up)["link"]
    except Exception as e:
        return {"status":"error","error":f"upload failed: {e}"}

    return {"status":"done","video_url":link}

runpod.serverless.start({"handler": handler})
