# OpenJutsu

Open-source Motion Transfer, Object Swap and Restyle for video, the features of Higgsfield Genjutsu, rebuilt on **open-weights models only** and running on your own ComfyUI.

> Entry for Comfy's **#OpenSourceHiggsfield** challenge, **Open Weights Only** category (https://comfy.org/open-source-higgsfield/). It uses no partner nodes, API nodes or Comfy Router calls. Every model is downloadable open weights.

## Modes
| Mode | Workflow | Models | What it does |
|---|---|---|---|
| Recast | `workflows/recast_wan22_animate.json` | Wan 2.2 Animate 14B, DWPose | Copies the motion from your clip onto a character from a reference image |
| Swap | `workflows/swap_wan_vace.json` | Wan 2.1 VACE 14B + SAM 2.1 mask | Replaces the object you clicked; the rest of the frame stays as filmed |
| Restyle | `workflows/restyle_wan_vace_depth.json` | Wan 2.1 VACE 14B, Depth-Anything-V2 | Redraws the whole clip in a new style and keeps the motion and camera |

`presets/` holds 13 one-click presets (anime, claymation, Lego, product swap, outfit swap, 8mm and more). To add one, drop in a new JSON file.

See **[docs/SPARK_HANDOFF.md](docs/SPARK_HANDOFF.md)** for the DGX Spark bring-up checklist.

## Architecture
Browser UI → FastAPI (`app/`): upload, SAM 2 click-to-mask, preset → patch workflow JSON → ComfyUI `/prompt` (`COMFY_URL`) → `/history` + `/view` for the result.

## Required ComfyUI custom node packs
- **comfyui_controlnet_aux** (Fannovel16) provides `DWPreprocessor` and `DepthAnythingV2Preprocessor`.
- Everything else is **ComfyUI core** (recent build, Oct 2026): `UNETLoader, CLIPLoader(type=wan), VAELoader, CLIPVisionLoader, CLIPVisionEncode, ModelSamplingSD3, CLIPTextEncode, LoadVideo, GetVideoComponents, LoadImage, ImageToMask, GrowMask, WanAnimateToVideo, WanVaceToVideo, TrimVideoLatent, KSampler, VAEDecode, CreateVideo, SaveVideo`.
- SAM 2 runs inside the app, not as a node. That way the mask is a plain mp4 and the workflow only needs core nodes.

### Verification status
- `WanAnimateToVideo` / `WanVaceToVideo` / `TrimVideoLatent` inputs and outputs were checked against ComfyUI master `comfy_extras/nodes_wan.py`.
- **Unverified (test on first run):**
  - the exact input names and enum values for `DWPreprocessor` / `DepthAnythingV2Preprocessor`
  - the `LoadVideo`/`SaveVideo`/`CreateVideo` widget values
  - `GetVideoComponents` output index 2 = fps
  - the sampler settings (steps, cfg, shift) are untuned starting values

## Setup

### DGX Spark (GB10, aarch64, 128 GB unified memory)
```bash
# 1. ComfyUI. Use NVIDIA's aarch64 CUDA PyTorch (or the NGC PyTorch container) and check that torch.cuda.is_available()
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI && cd ~/ComfyUI
pip install -r requirements.txt
git clone https://github.com/Fannovel16/comfyui_controlnet_aux custom_nodes/comfyui_controlnet_aux
pip install -r custom_nodes/comfyui_controlnet_aux/requirements.txt   # onnxruntime: use onnxruntime-gpu aarch64 build or CPU fallback
python main.py --listen 0.0.0.0 --port 8188
# 2. models (bf16 fits comfortably in 128 GB)
cd /path/to/openjutsu && COMFYUI_DIR=~/ComfyUI ./scripts/download_models.sh
# 3. app
pip install -r requirements.txt && pip install "git+https://github.com/facebookresearch/sam2.git"
COMFY_URL=http://127.0.0.1:8188 uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Notes for aarch64: SAM 2's optional CUDA extension may fail to build. If it does, set `SAM2_BUILD_CUDA=0`; the extension only does mask post-processing. DWPose's onnxruntime-gpu may not have an aarch64 wheel; the CPU fallback works but is slower.

### 24 GB GPUs (RTX 4090/3090/5090)
Run `./scripts/download_models.sh --fp8` (int8 Animate) and set `unet_name` in the recast workflow to match. For VACE, use an fp8/GGUF quant, or run ComfyUI with `--lowvram` / `--reserve-vram 2`. Keep clips around 480p and 81 frames or shorter.

## Models (all open weights, Hugging Face)
- `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`: `wan2.2_animate_14B_bf16` (or `_int8_convrot`), `umt5_xxl_fp8_e4m3fn_scaled`, `wan_2.1_vae`
- `Comfy-Org/Wan_2.1_ComfyUI_repackaged`: `wan2.1_vace_14B_fp16`, `clip_vision_h`
- `facebook/sam2.1-hiera-large` (SAM 2.1)
- DWPose and Depth-Anything-V2 (downloaded automatically by controlnet_aux)

Check each model's license before you redistribute outputs. The Wan models are Apache-2.0 and SAM 2 is Apache-2.0.

## Comfy Developer Platform
The contest requires Comfy API, Comfy Router and/or Comfy SDK. The workflows are API-format JSON, so they can be deployed as-is with **Comfy API**. Wiring in the **Comfy SDK** is a TODO before submission. Router is deliberately not used, to stay Open-Weights-Only.

## License
Apache-2.0. See LICENSE.
