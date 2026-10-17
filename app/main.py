"""OpenJutsu web app: upload a clip, click to mask (SAM2), pick a preset, queue on ComfyUI.
Only talks to a ComfyUI server you run (COMFY_URL). No partner/API nodes are used."""
import json, os, uuid, shutil, time, pathlib
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
WORK = pathlib.Path(os.environ.get("OPENJUTSU_WORK", ROOT / "work")); WORK.mkdir(exist_ok=True)
app = FastAPI(title="OpenJutsu")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

def presets():
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "presets").glob("*.json"))}

@app.get("/")
def index(): return FileResponse(ROOT / "app" / "static" / "index.html")

@app.get("/api/presets")
def list_presets(): return list(presets().values())

@app.post("/api/upload")
async def upload(video: UploadFile = File(...)):
    sid = uuid.uuid4().hex[:12]; d = WORK / sid; d.mkdir()
    src = d / "input.mp4"
    with src.open("wb") as f: shutil.copyfileobj(video.file, f)
    import cv2
    cap = cv2.VideoCapture(str(src)); ok, frame = cap.read(); cap.release()
    if not ok: raise HTTPException(400, "could not read video")
    cv2.imwrite(str(d / "frame0.jpg"), frame)
    return {"session": sid, "width": frame.shape[1], "height": frame.shape[0]}

@app.get("/api/frame/{sid}")
def frame(sid: str): return FileResponse(WORK / sid / "frame0.jpg")

@app.post("/api/mask")
def mask(sid: str = Form(...), points: str = Form(...)):
    """points: JSON [[x,y,label],...] on frame 0 in pixels. Propagates through the clip."""
    from app.sam2_mask import propagate_mask
    d = WORK / sid
    pts = json.loads(points)
    propagate_mask(d / "input.mp4", pts, d / "mask.mp4", d / "mask_preview.png")
    return {"preview": f"/api/maskpreview/{sid}?t={time.time()}"}

@app.get("/api/maskpreview/{sid}")
def maskpreview(sid: str): return FileResponse(WORK / sid / "mask_preview.png")

async def comfy_upload(client, path: pathlib.Path, name: str):
    r = await client.post(f"{COMFY}/upload/image", files={"image": (name, path.read_bytes())},
                          data={"overwrite": "true", "type": "input"})
    r.raise_for_status(); j = r.json()
    return (j.get("subfolder") + "/" if j.get("subfolder") else "") + j["name"]

@app.post("/api/run")
async def run(sid: str = Form(...), preset: str = Form(...), prompt: str = Form(""),
              seed: int = Form(0), reference: UploadFile | None = File(None)):
    p = presets().get(preset) or HTTPException(404, "preset")
    if isinstance(p, HTTPException): raise p
    d = WORK / sid
    wf = json.loads((ROOT / "workflows" / f"{p['workflow']}.json").read_text())
    async with httpx.AsyncClient(timeout=120) as c:
        wf["10"]["inputs"]["file"] = await comfy_upload(c, d / "input.mp4", f"oj_{sid}_input.mp4")
        if p["mode"] == "swap":
            if not (d / "mask.mp4").exists(): raise HTTPException(400, "click the object to mask first")
            wf["15"]["inputs"]["file"] = await comfy_upload(c, d / "mask.mp4", f"oj_{sid}_mask.mp4")
            wf["18"]["inputs"]["expand"] = p["params"].get("mask_expand", 8)
        if reference is not None and reference.filename:
            rp = d / "reference.png"; rp.write_bytes(await reference.read())
            wf["12"]["inputs"]["image"] = await comfy_upload(c, rp, f"oj_{sid}_ref.png")
        elif p.get("requires_reference_image"):
            raise HTTPException(400, "this preset needs a reference image")
        else:  # drop optional reference branch
            for k in ("12", "13", "14"): wf.pop(k, None)
            for k in ("reference_image", "clip_vision_output"): wf["30"]["inputs"].pop(k, None)
        wf["5"]["inputs"]["text"] = (prompt + ", " if prompt else "") + p["prompt"]
        ks = wf["40"]["inputs"]; ks["seed"] = seed; ks["steps"] = p["params"]["steps"]; ks["cfg"] = p["params"]["cfg"]
        if "strength" in wf["30"]["inputs"]: wf["30"]["inputs"]["strength"] = p["params"]["strength"]
        wf["44"]["inputs"]["filename_prefix"] = f"openjutsu/{sid}"
        r = await c.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": sid})
        if r.status_code != 200: raise HTTPException(502, r.text)
        return r.json()

@app.get("/api/status/{prompt_id}")
async def status(prompt_id: str):
    async with httpx.AsyncClient(timeout=30) as c:
        h = (await c.get(f"{COMFY}/history/{prompt_id}")).json()
    if prompt_id not in h: return {"done": False}
    outs = []
    for node in h[prompt_id].get("outputs", {}).values():
        for key in ("images", "videos", "gifs"):
            for f in node.get(key, []):
                outs.append(f"/api/view?filename={f['filename']}&subfolder={f.get('subfolder','')}&type={f.get('type','output')}")
    return {"done": True, "status": h[prompt_id].get("status", {}), "outputs": outs}

@app.get("/api/view")
async def view(filename: str, subfolder: str = "", type: str = "output"):
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.get(f"{COMFY}/view", params={"filename": filename, "subfolder": subfolder, "type": type})
    return Response(r.content, media_type=r.headers.get("content-type", "video/mp4"))
