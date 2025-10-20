from fastapi import FastAPI, Request
import os, subprocess, uuid, json, pathlib

app = FastAPI()

@app.post("/render")
async def render(req: Request):
    data = await req.json()
    image_url = data["image_url"]
    audio_url = data["audio_url"]
    quality = data.get("quality", "720p")

    job_id = str(uuid.uuid4())
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Download the input files
    os.system(f"wget -q -O input.png '{image_url}'")
    os.system(f"wget -q -O input.wav '{audio_url}'")

    # Save request info for the model
    with open("request.json", "w") as f:
        json.dump({"image": "input.png", "audio": "input.wav"}, f)

    # Choose size config
    size_flag = "infinitetalk-720" if quality in ["720p", "1080p"] else "infinitetalk-480"

    # Run InfiniteTalk generation
    cmd = (
        "python /app/InfiniteTalk/generate_infinitetalk.py "
        "--ckpt_dir /app/weights/Wan2.1-I2V-14B-480P "
        "--wav2vec_dir /app/weights/chinese-wav2vec2-base "
        "--infinitetalk_dir /app/weights/InfiniteTalk/single/infinitetalk.safetensors "
        "--input_json request.json "
        f"--size {size_flag} --sample_steps 40 --mode streaming --motion_frame 9 --save_file out_"
    )
    subprocess.run(cmd, shell=True, check=True)

    # Find generated video
    mp4_file = next(pathlib.Path('.').glob('out_*.mp4'), None)
    if not mp4_file:
        return {"status": "error", "error": "No MP4 file generated"}

    # Optional upscale to 1080p
    if quality == "1080p":
        subprocess.run(
            f"ffmpeg -y -i {mp4_file} -vf scale=1920:1080:flags=lanczos -c:v libx264 -preset medium -crf 17 -c:a copy out_1080p.mp4",
            shell=True,
            check=True
        )
        mp4_file = pathlib.Path("out_1080p.mp4")

    # Upload to a free file host for quick URL access (temporary)
    upload_output = subprocess.check_output(
        f"curl -s -F 'file=@{mp4_file}' https://file.io", shell=True
    ).decode()

    try:
        video_url = json.loads(upload_output)["link"]
    except:
        video_url = None

    return {
        "job_id": job_id,
        "status": "done",
        "video_url": video_url
    }

@app.get("/status/{job_id}")
async def status(job_id: str):
    return {"job_id": job_id, "status": "done"}
