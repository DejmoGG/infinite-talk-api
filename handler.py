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
    """RunPod job entrypoint.
       Input:
       {
         "input": {
           "image_url": "https://...png|jpg",
           "audio_url": "https://...mp3|wav",
           "quality": "720p"|"1080p"
         }
       }
    """
    i = event.get("input") or {}
    image_url = i.get("image_url")
    audio_url = i.get("audio_url")
    quality   = (i.get("quality") or "720p").lower()

    if not image_url or not audio_url:
        return {"status": "error", "error": "image_url and audio_url are required"}

    work = pathlib.Path(f"/tmp/{uuid.uuid4().hex}")
    work.mkdir(parents=True, exist_ok=True)

    img_path = work / "input.png"
    raw_audio = work / "raw_audio"
    wav_path = work / "input.wav"
    out_path = work / "out.mp4"

    # 1) download inputs
    _download(image_url, str(img_path))
    _download(audio_url, str(raw_audio))

    # 2) normalize audio for the pipeline
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_audio), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": f"ffmpeg convert failed: {e.stdout.decode('utf-8', 'ignore')[:4000]}"}

    # 3) run InfiniteTalk (your repo has generate_infinitetalk.py at top-level)
    res = "1280x720" if quality == "720p" else "1920x1080"
    cmd = [
        "python", "generate_infinitetalk.py",
        "--image", str(img_path),
        "--audio", str(wav_path),
        "--output", str(out_path),
        "--resolution", res
    ]
    try:
        subprocess.run(cmd, cwd="/app", check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": f"InfiniteTalk failed: {e.stdout.decode('utf-8', 'ignore')[:4000]}"}

    if not out_path.exists():
        return {"status": "error", "error": "No output video produced"}

    # 4) upload to file.io and return link
    try:
        up = subprocess.check_output(["curl", "-sF", f"file=@{out_path}", "https://file.io"]).decode("utf-8")
        link = json.loads(up)["link"]
    except Exception as e:
        return {"status": "error", "error": f"Upload failed: {str(e)} / {up[:4000] if 'up' in locals() else ''}"}

    return {"status": "done", "video_url": link}

runpod.serverless.start({"handler": handler})
