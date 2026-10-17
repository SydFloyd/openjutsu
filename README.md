# OpenJutsu

Open-source Motion Transfer, Object Swap and Restyle for video, the features of Higgsfield Genjutsu, rebuilt on **open-weights models only** and running on your own ComfyUI.

> Entry for Comfy's **#OpenSourceHiggsfield** challenge, **Open Weights Only** category (https://comfy.org/open-source-higgsfield/). It uses no partner nodes, API nodes or Comfy Router calls. Every model is downloadable open weights.

## Modes
| Mode | Workflow | Models | What it does |
|---|---|---|---|
| Recast | `workflows/recast_wan22_animate.json` | Wan 2.2 Animate 14B, DWPose | Copies the motion from your clip onto a character from a reference image |
| Swap | `workflows/swap_wan_vace.json` | Wan 2.1 VACE 14B + SAM 2.1 mask | Replaces a clicked object using a tracked mask and a product reference |
| Restyle | `workflows/restyle_wan_vace_depth.json` | Wan 2.1 VACE 14B, Depth-Anything-V2 | Redraws the whole clip in a new style and keeps the motion and camera |

`presets/` holds 13 one-click presets (anime, claymation, Lego, product swap, outfit swap, 8mm and more). To add one, drop in a new JSON file.

For Recast, choose one clearly visible person and a reference with similar framing and body proportions. This graph drives body/hand poses; it does not implement pose retargeting or a separate facial-expression video. A mismatched full-body cartoon reference produced a duplicate character in testing. See the report and [Wan's preprocessing guidance](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md) before using very different proportions.

See **[docs/SPARK_HANDOFF.md](docs/SPARK_HANDOFF.md)** for the DGX Spark bring-up checklist.

## Architecture
Browser UI → FastAPI (`app/`): upload, SAM 2 click-to-mask, preset → **official `comfy-sdk`** → **local `comfy-api-proxy`** → your ComfyUI. The app downloads the completed videos through the SDK and serves them to the browser.

Uploads are normalized to 832×480 at 16 fps. Longer clips use bounded windows: Recast carries five generated frames into the next window; VACE uses nine overlapping frames and a crossfade. Final padding is removed, and the original audio is restored. App jobs retain their child Comfy job IDs and report failures to the UI.

## Required ComfyUI custom node packs
- **comfyui_controlnet_aux** (Fannovel16) provides `DWPreprocessor` and `DepthAnythingV2Preprocessor`.
- Everything else is **ComfyUI core** (recent build, Oct 2026): `UNETLoader, CLIPLoader(type=wan), VAELoader, CLIPVisionLoader, CLIPVisionEncode, ModelSamplingSD3, CLIPTextEncode, LoadVideo, GetVideoComponents, LoadImage, ImageToMask, ThresholdMask, GrowMask, EmptyImage, ImageCompositeMasked, WanAnimateToVideo, WanVaceToVideo, TrimVideoLatent, KSampler, VAEDecode, CreateVideo, SaveVideo`.
- SAM 2 runs inside the app. Masks are binary **lossless FFV1 MKV**, loaded with core video nodes. Both the original Meta `.pt` and Kijai's `.safetensors` repack are supported.
- Swap fills the expanded edit region with RGB 127 before VACE conditioning, following [VACE's inpainting input format](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md). The mask still marks that region for generation.

### Verification status
- All three graphs are checked against live `/object_info`, including filenames, widgets, links, and the absence of partner/API nodes.
- `GetVideoComponents` slot 2 is fps. `CLIPVisionEncode.crop=none`, `ImageToMask.channel=red`, and `CreateVideo.images/fps` are valid.
- DWPose uses both CUDA TorchScript models; DepthAnything uses the Apache-2.0 Small checkpoint. SaveVideo uses MP4 and its default automatic codec.
- Reproduce the schema check with `COMFY_URL=http://127.0.0.1:8191 python scripts/validate_workflows.py`.
- Measured runs, quality findings and remaining limits are recorded in [the Spark report](docs/SPARK_BRINGUP.md).

## Setup

### DGX Spark (GB10, aarch64, 128 GB unified memory)
```bash
# 1. ComfyUI. Use NVIDIA's aarch64 CUDA PyTorch (or the NGC PyTorch container) and check that torch.cuda.is_available()
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI && cd ~/ComfyUI
pip install -r requirements.txt
git clone https://github.com/Fannovel16/comfyui_controlnet_aux custom_nodes/comfyui_controlnet_aux
pip install -r custom_nodes/comfyui_controlnet_aux/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
python main.py --listen 0.0.0.0 --port 8191 --offline --use-pytorch-cross-attention
# 2. models (bf16 fits comfortably in 128 GB)
cd /path/to/openjutsu && COMFYUI_DIR=~/ComfyUI ./scripts/download_models.sh
```
Install a CUDA-enabled aarch64 PyTorch build **before** the ComfyUI requirements. On an existing Spark install, reuse its verified runtime. `--offline` disables partner nodes; model files must be downloaded during setup. If controlnet_aux's first-run downloads are needed, do that setup online before enabling offline operation.

### App and official SDK proxy

Run these on the laptop or another CPU host; Spark only needs ComfyUI. For temporary same-machine testing, the app can reuse Spark's CUDA environment. SAM2 uses CUDA when available and otherwise runs on CPU. Install an appropriate PyTorch/torchvision build on the app host first.

```bash
pip install -r requirements.txt
SAM2_BUILD_CUDA=0 pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
./scripts/download_models.sh --sam2-only

# Terminal 1: use the Spark LAN address instead of loopback when on the laptop.
comfy-api-proxy run --comfyui http://127.0.0.1:8191 --host 127.0.0.1 --port 8189

# Terminal 2: this is the local proxy URL, not the ComfyUI URL.
COMFY_BASE_URL=http://127.0.0.1:8189 uvicorn app.main:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`. The proxy needs no Comfy Cloud key. Keep the app and proxy running while jobs execute; interrupted jobs are not automatically resumed after a restart. `SAM2_CKPT` can point to an existing checkpoint; `SAM2_CFG` defaults to `configs/sam2.1/sam2.1_hiera_l.yaml`. The optional SAM2 CUDA extension is disabled, including mask post-processing that depends on it.

On ARM, decord is unnecessary: the app uses PyAV and bundled ARM ffmpeg. DWPose uses `yolox_l.torchscript.pt` plus `dw-ll_ucoco_384_bs5.torchscript.pt`, so GPU ONNX Runtime is not required. SageAttention and xformers are optional; the tested setup uses PyTorch SDPA.

CPU checks: `pip install -r requirements-dev.txt && python -m pytest -q`.

### 24 GB GPUs (RTX 4090/3090/5090)
This configuration was not benchmarked during the Spark bring-up.

Run `./scripts/download_models.sh --fp8` (int8 Animate) and set `unet_name` in the recast workflow to match. For VACE, use an fp8/GGUF quant, or run ComfyUI with `--lowvram` / `--reserve-vram 2`. Keep clips around 480p and 81 frames or shorter.

## Models (all open weights, Hugging Face)
- `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`: `wan2.2_animate_14B_bf16` (or `_int8_convrot`)
- `Comfy-Org/Wan_2.1_ComfyUI_repackaged`: `wan2.1_vace_14B_fp16`, `umt5_xxl_fp16`, `clip_vision_h`
- `Kijai/WanVideo_comfy`: `Wan2_1_VAE_bf16` (the Wan 2.1 VAE serves both 14B models)
- `Kijai/sam2-safetensors`: `sam2.1_hiera_large-fp16` (Meta SAM 2.1 Hiera Large repack)
- DWPose TorchScript and DepthAnything V2 **Small**, downloaded by controlnet_aux during online setup

Check each model's license before you redistribute outputs. The Wan models are Apache-2.0 and SAM 2 is Apache-2.0.

## Comfy Developer Platform
The app uses the official **Comfy SDK workflow surface** against the self-hosted **Comfy API v2 proxy**. It binds video/image assets with `client.assets.from_file`, submits API-format graphs with `client.submit`, follows job status, and downloads node 44's outputs. It does not call the SDK's model-router surface or partner nodes. See [official SDK documentation](https://docs.comfy.org/development/api-development/sdks).

This setup does not create a public hosted endpoint. The local proxy and app can run on the laptop while ComfyUI inference stays on Spark.

## License
Apache-2.0. See LICENSE.
