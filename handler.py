# handler.py
import os, uuid, json, pathlib, subprocess, requests
import runpod

def _download(url: str, path: str):
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 14):
            if chunk:
                f.write(chunk)

def handler(event):
    """
    Input JSON:
    {
      "input": {
        "image_url": "https://...png",
        "audio_url": "https://...wav",
        "quality": "720p" | "1080p"
      }
    }
    """
    i = event.get("input") or {}
    image_url = i.get("image_url")
    audio_url = i.get("audio_url")
    quality   = (i.get("quality") or "720p").lower()

    if not image_url or not audio_url:
        return {"status": "error", "error": "image_url and audio_url are required"}

    # Prepare workspace
    job_id = uuid.uuid4().hex
    work = pathlib.Path(f"/tmp/{job_id}")
    work.mkdir(parents=True, exist_ok=True)

    img_path = work / "input.png"
    raw_audio = work / "raw_audio"
    wav_path = work / "input.wav"

    # 1️⃣ Download inputs
    _download(image_url, str(img_path))
    _download(audio_url, str(raw_audio))

    # 2️⃣ Normalize audio to 16kHz mono WAV
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_audio),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": f"ffmpeg convert failed: {e.stdout.decode()[:4000]}"}

    # 3️⃣ Build request.json
    request_json = work / "request.json"
    with open(request_json, "w") as f:
        json.dump({"image": str(img_path), "audio": str(wav_path), "seed": 1}, f)

    # 4️⃣ Run InfiniteTalk
    size_flag = "infinitetalk-720" if quality in ["720p", "1080p"] else "infinitetalk-480"
    cmd = [
        "python", "/app/InfiniteTalk/generate_infinitetalk.py",
        "--ckpt_dir", "/app/weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "/app/weights/chinese-wav2vec2-base",
        "--infinitetalk_dir", "/app/weights/InfiniteTalk/single/infinitetalk.safetensors",
        "--input_json", str(request_json),
        "--size", size_flag,
        "--sample_steps", "40",
        "--mode", "streaming",
        "--motion_frame", "9",
        "--save_file", "out_"
    ]

    try:
        subprocess.run(cmd, cwd=work, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": f"InfiniteTalk failed: {e.stdout.decode()[:4000]}"}

    # 5️⃣ Find output video
    mp4_files = list(work.glob("out_*.mp4"))
    if not mp4_files:
        return {"status": "error", "error": "No output video produced"}
    mp4_path = mp4_files[0]

    # Optional upscale to 1080p
    if quality == "1080p":
        upscale_path = work / "out_1080p.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(mp4_path),
            "-vf", "scale=1920:1080:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-c:a", "copy",
            str(upscale_path)
        ], check=True)
        mp4_path = upscale_path

    # 6️⃣ Upload to file.io
    try:
        up = subprocess.check_output(["curl", "-sF", f"file=@{mp4_path}", "https://file.io"]).decode("utf-8")
        video_url = json.loads(up)["link"]
    except Exception as e:
        return {"status": "error", "error": f"Upload failed: {e}"}

    return {"status": "done", "video_url": video_url}

# Entrypoint
runpod.serverless.start({"handler": handler})
