from fastapi import FastAPI, Request
import os, subprocess, uuid, json, pathlib

app = FastAPI()

# --- Helper ---------------------------------------------------------
def run(cmd: str):
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

# --- API ------------------------------------------------------------
@app.post("/render")
async def render(req: Request):
    """
    INPUT  (JSON):
    {
      "image_url": "https://.../face.png",
      "audio_url": "https://.../voice.wav",
      "quality": "720p" | "1080p"  (default: 720p)
    }

    OUTPUT (JSON):
    {
      "job_id": "uuid",
      "status": "done" | "error",
      "video_url": "https://.../file.mp4",   # present when status == "done"
      "error": "message"                     # present when status == "error"
    }
    """
    data = await req.json()
    image_url = data["image_url"]
    audio_url = data["audio_url"]
    quality   = data.get("quality", "720p")

    job_id = str(uuid.uuid4())
    work = pathlib.Path(f"/tmp/{job_id}")
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)

    # 1) Download inputs
    run(f"wget -q -O input.png '{image_url}'")
    run(f"wget -q -O input.wav '{audio_url}'")

    # 2) Build request json for InfiniteTalk
    with open("request.json","w") as f:
        json.dump({"image":"input.png","audio":"input.wav","seed":1}, f)

    # 3) Run InfiniteTalk (native 480p/720p)
    size_flag = "infinitetalk-720" if quality in ["720p","1080p"] else "infinitetalk-480"
    gen = (
        "python /app/InfiniteTalk/generate_infinitetalk.py "
        "--ckpt_dir /app/weights/Wan2.1-I2V-14B-480P "
        "--wav2vec_dir /app/weights/chinese-wav2vec2-base "
        "--infinitetalk_dir /app/weights/InfiniteTalk/single/infinitetalk.safetensors "
        "--input_json request.json "
        f"--size {size_flag} --sample_steps 40 --mode streaming --motion_frame 9 "
        "--save_file out_"
    )
    run(gen)

    # 4) Find output
    mp4 = next((p for p in pathlib.Path('.').glob('out_*.mp4')), None)
    if not mp4:
        return {"job_id": job_id, "status": "error", "error": "No MP4 generated"}

    # 5) Optional upscale to 1080p (simple, reliable)
    if quality == "1080p":
        run(f"ffmpeg -y -i {mp4} -vf scale=1920:1080:flags=lanczos "
            f"-c:v libx264 -preset medium -crf 17 -c:a copy out_1080p.mp4")
        mp4 = pathlib.Path("out_1080p.mp4")

    # 6) Upload to get a public URL (temporary host, perfect for MVP)
    #    Swap to S3/CDN later if needed.
    up = subprocess.check_output(
        f"curl -s -F 'file=@{mp4}' https://file.io", shell=True
    ).decode()

    try:
        video_url = json.loads(up)["link"]
    except Exception as e:
        return {"job_id": job_id, "status": "error", "error": f"Upload failed: {e}"}

    return {"job_id": job_id, "status": "done", "video_url": video_url}

@app.get("/status/{job_id}")
def status(job_id: str):
    # We run synchronously and return the URL in /render, so status is trivial.
    return {"job_id": job_id, "status": "done"}
