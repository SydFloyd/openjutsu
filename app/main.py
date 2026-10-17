"""OpenJutsu: local uploads and SAM2, official Comfy SDK workflow execution."""
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from comfy_sdk import AsyncComfy
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import video

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("OPENJUTSU_WORK", ROOT / "work")).resolve()
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "jobs").mkdir(exist_ok=True)
# Without this setting the SDK defaults to Comfy Cloud.
os.environ["COMFY_BASE_URL"] = os.environ.get("COMFY_BASE_URL", "").strip() or "http://127.0.0.1:8189"
app = FastAPI(title="OpenJutsu")
app.mount("/static", StaticFiles(directory=ROOT / "app/static"), name="static")
generation_lock = asyncio.Lock()


def presets():
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "presets").glob("*.json"))}


def session(sid):
    if not re.fullmatch(r"[0-9a-f]{12}", sid) or not (WORK / sid / "input.mp4").is_file():
        raise HTTPException(404, "Unknown upload session")
    return WORK / sid


def job_path(job_id):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "Unknown job")
    return WORK / "jobs" / f"{job_id}.json"


def save_job(job_id, state):
    path = job_path(job_id)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


@app.get("/")
def index():
    return FileResponse(ROOT / "app/static/index.html")


@app.get("/api/presets")
def list_presets():
    return list(presets().values())


@app.post("/api/upload")
async def upload(video_file: UploadFile = File(..., alias="video")):
    sid = uuid.uuid4().hex[:12]
    directory = WORK / sid
    directory.mkdir()
    source = directory / "original.mp4"
    with source.open("wb") as output:
        await asyncio.to_thread(shutil.copyfileobj, video_file.file, output)
    try:
        info = await asyncio.to_thread(video.normalize, source, directory / "input.mp4")
        await asyncio.to_thread(video.ffmpeg, "-i", directory / "input.mp4", "-frames:v", 1, directory / "frame0.jpg")
    except (ValueError, OSError) as error:
        shutil.rmtree(directory)
        raise HTTPException(400, f"Could not read video: {error}") from error
    return {"session": sid, **info}


@app.get("/api/frame/{sid}")
def frame(sid: str):
    return FileResponse(session(sid) / "frame0.jpg")


@app.post("/api/mask")
def mask(sid: str = Form(...), points: str = Form(...)):
    from app.sam2_mask import propagate_mask
    directory = session(sid)
    try:
        result = propagate_mask(directory / "input.mp4", json.loads(points), directory / "mask.mkv", directory / "mask_preview.png")
    except (ValueError, TypeError) as error:
        raise HTTPException(400, str(error)) from error
    return {"preview": f"/api/maskpreview/{sid}?t={time.time()}", **result}


@app.get("/api/maskpreview/{sid}")
def maskpreview(sid: str):
    return FileResponse(session(sid) / "mask_preview.png")


def workflow(preset, prompt, seed, length, steps=None, cfg=None, shift=None):
    graph = json.loads((ROOT / "workflows" / f"{preset['workflow']}.json").read_text())
    params = preset["params"]
    graph["5"]["inputs"]["text"] = (prompt + ", " if prompt else "") + preset["prompt"]
    graph["40"]["inputs"].update(seed=seed, steps=steps or params["steps"], cfg=params["cfg"] if cfg is None else cfg)
    graph["4"]["inputs"]["shift"] = params.get("shift", 8.0) if shift is None else shift
    graph["30"]["inputs"]["length"] = length
    if "strength" in graph["30"]["inputs"]:
        graph["30"]["inputs"]["strength"] = params["strength"]
    if preset["mode"] == "swap":
        graph["18"]["inputs"]["expand"] = params.get("mask_expand", 8)
    return graph


async def generate(job_id, directory, preset, prompt, seed, reference, total, capacity, steps, cfg, shift):
    state = {"done": False, "status": "queued", "outputs": [], "comfy_jobs": []}
    async with generation_lock:
        started = time.monotonic()
        mode = preset["mode"]
        plan = list(video.windows(total, capacity, 5 if mode == "recast" else 9))
        state.update(status="running", chunks=len(plan), completed_chunks=0)
        save_job(job_id, state)
        output_dir = directory / job_id
        output_dir.mkdir(exist_ok=True)
        chunks = []
        try:
            async with AsyncComfy(timeout=120) as client:
                for index, window in enumerate(plan):
                    graph = workflow(preset, prompt, seed, window["length"], steps, cfg, shift)
                    inputs = {("10", "file"): directory / "input.mp4"}
                    if mode == "recast":
                        graph["30"]["inputs"]["video_frame_offset"] = window["start"] + window["overlap"]
                        if chunks:
                            continuation = output_dir / "continue.mkv"
                            tail = await asyncio.to_thread(video.read_frames, chunks[-1])
                            await asyncio.to_thread(video.write_frames, tail[-5:], continuation, lossless=True)
                            graph["50"] = {"class_type": "LoadVideo", "inputs": {"file": "continue.mkv"}}
                            graph["51"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}
                            graph["30"]["inputs"]["continue_motion"] = ["51", 0]
                            inputs[("50", "file")] = continuation
                    else:
                        source = output_dir / f"input-{index}.mp4"
                        await asyncio.to_thread(video.slice_video, directory / "input.mp4", source, window["start"], window["count"], window["length"])
                        inputs[("10", "file")] = source
                        if mode == "swap":
                            mask_file = output_dir / f"mask-{index}.mkv"
                            await asyncio.to_thread(video.slice_video, directory / "mask.mkv", mask_file, window["start"], window["count"], window["length"], True)
                            inputs[("15", "file")] = mask_file
                    if reference:
                        inputs[("12", "image")] = reference
                    else:
                        for node in ("12", "13", "14"):
                            graph.pop(node, None)
                        for key in ("reference_image", "clip_vision_output"):
                            graph["30"]["inputs"].pop(key, None)
                    graph["44"]["inputs"]["filename_prefix"] = f"openjutsu/{job_id}/chunk-{index}"
                    wf = client.workflows.from_json(graph)
                    for (node, field), path in inputs.items():
                        wf.set_input(node, field, client.assets.from_file(path))
                    job = await client.submit(wf)
                    state["comfy_jobs"].append(job.id)
                    save_job(job_id, state)
                    await job.wait(timeout=14400)
                    if job.status != "succeeded":
                        raise RuntimeError(f"ComfyUI {job.status}: {job.error}")
                    outputs = job.get_outputs("44")
                    if not outputs:
                        raise RuntimeError("ComfyUI completed without a video output")
                    destination = output_dir / f"chunk-{index}.mp4"
                    await outputs[0].to_file(destination)
                    chunks.append(destination)
                    state["completed_chunks"] = len(chunks)
                    save_job(job_id, state)
            await asyncio.to_thread(video.assemble, chunks, plan, output_dir / "result.mp4", total, mode, directory / "original.mp4")
            state.update(done=True, status="succeeded", outputs=[f"/api/result/{job_id}"],
                         session=directory.name, result_frames=total, fps=video.FPS,
                         wall_seconds=time.monotonic() - started)
        except Exception as error:
            state.update(done=True, status="failed", error=str(error), wall_seconds=time.monotonic() - started)
        save_job(job_id, state)


@app.post("/api/run")
async def run(background_tasks: BackgroundTasks, sid: str = Form(...), preset: str = Form(...),
              prompt: str = Form(""), seed: int = Form(0), reference: UploadFile | None = File(None),
              max_frames: int | None = Form(None, gt=0), chunk_frames: int | None = Form(None),
              steps: int | None = Form(None, gt=0), cfg: float | None = Form(None), shift: float | None = Form(None)):
    selected = presets().get(preset)
    if selected is None:
        raise HTTPException(404, "Unknown preset")
    directory = session(sid)
    if selected["mode"] == "swap" and not (directory / "mask.mkv").exists():
        raise HTTPException(400, "Click the object and track its mask first")
    job_id = uuid.uuid4().hex
    reference_path = None
    if reference is not None and reference.filename:
        reference_path = directory / f"reference-{job_id}.png"
        reference_path.write_bytes(await reference.read())
    elif selected.get("requires_reference_image"):
        raise HTTPException(400, "This preset needs a reference image")
    total = video.probe(directory / "input.mp4")["frames"]
    if max_frames is not None:
        total = min(total, max_frames)
    capacity = chunk_frames or (77 if selected["mode"] == "recast" else 81)
    try:
        list(video.windows(total, capacity, 5 if selected["mode"] == "recast" else 9))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    save_job(job_id, {"done": False, "status": "queued", "outputs": [], "comfy_jobs": []})
    background_tasks.add_task(generate, job_id, directory, selected, prompt, seed, reference_path, total, capacity, steps, cfg, shift)
    return {"prompt_id": job_id, "status": "queued"}


@app.get("/api/status/{prompt_id}")
def status(prompt_id: str):
    path = job_path(prompt_id)
    if not path.exists():
        raise HTTPException(404, "Unknown job")
    return json.loads(path.read_text())


@app.get("/api/result/{prompt_id}")
def result(prompt_id: str):
    state = status(prompt_id)
    if state["status"] != "succeeded":
        raise HTTPException(409, state.get("error", "Generation is still running"))
    return FileResponse(session(state["session"]) / prompt_id / "result.mp4", media_type="video/mp4")
