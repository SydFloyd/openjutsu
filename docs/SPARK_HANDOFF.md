# DGX Spark handoff: bring-up checklist for OpenJutsu

You are the agent configuring the **DGX Spark** (GB10, aarch64, 128 GB unified memory). The models are **already downloaded**. Your job is to get all three workflows running end to end, fix whatever doesn't work, and report back.

**Hard deadline:** the contest submission closes **Fri Oct 30, 12:00 PM ET** (9am PT). Aim to finish by Oct 29.
**Hard rule (Open Weights Only):** no partner nodes, API nodes or Comfy Router calls, and no closed models. Every model must be open weights. Any substitution you make must also be open weights.

## 0. Environment
- [ ] `python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"` → CUDA must be True. If not, install NVIDIA's aarch64 CUDA PyTorch build (or use the NGC PyTorch container). Do not use the CPU wheel from PyPI.
- [ ] ComfyUI is on a recent master (it needs `WanAnimateToVideo`, `WanVaceToVideo`, `LoadVideo`, `CreateVideo`, `SaveVideo`). Start it with `python main.py --listen 0.0.0.0 --port 8188`.
- [ ] `custom_nodes/comfyui_controlnet_aux` is installed and loads without import errors (check the startup log).

## 1. Model filenames and paths
The workflows expect these names under `ComfyUI/models/`. Rename the files, or edit the workflow JSON, until they match:

| Folder | File | Used by |
|---|---|---|
| diffusion_models | `wan2.2_animate_14B_bf16.safetensors` | recast |
| diffusion_models | `wan2.1_vace_14B_fp16.safetensors` | swap, restyle |
| text_encoders | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | all |
| vae | `wan_2.1_vae.safetensors` | all |
| clip_vision | `clip_vision_h.safetensors` | recast |
| (repo) `models/sam2/` | `sam2.1_hiera_large.pt` | app (`SAM2_CKPT`) |

- [ ] `curl -s localhost:8188/object_info/UNETLoader | jq '.UNETLoader.input.required.unet_name[0]'` lists both diffusion models. Do the same for CLIPLoader, VAELoader and CLIPVisionLoader.
- [ ] **SAM2 checkpoint:** the name `sam2.1_hiera_large.pt` is unverified (the HF repo `facebook/sam2.1-hiera-large` may ship only the transformers format). If the file isn't there, download it from Meta's direct link (see the facebookresearch/sam2 README `checkpoints/download_ckpts.sh`). Set `SAM2_CKPT`/`SAM2_CFG` to match (`configs/sam2.1/sam2.1_hiera_l.yaml`).

## 2. Node and widget checks (`/object_info`)
For every class below, run `curl -s localhost:8188/object_info/<Class> | jq` and fix the workflow JSON wherever the names or values differ:

`UNETLoader, CLIPLoader, VAELoader, CLIPVisionLoader, CLIPVisionEncode, ModelSamplingSD3, CLIPTextEncode, LoadVideo, GetVideoComponents, LoadImage, ImageToMask, GrowMask, WanAnimateToVideo, WanVaceToVideo, TrimVideoLatent, KSampler, VAEDecode, CreateVideo, SaveVideo, DWPreprocessor, DepthAnythingV2Preprocessor`

The following were **not confirmed** when the workflows were written:
- [ ] `DWPreprocessor`: `bbox_detector` (`yolox_l.onnx`?), `pose_estimator` (`dw-ll_ucoco_384_bs5.torchscript.pt`?), `scale_stick_for_xinsr_cn` (does it exist? is the value `disable`?), `resolution`
- [ ] `DepthAnythingV2Preprocessor`: `ckpt_name` value (`depth_anything_v2_vitl.pth`?)
- [ ] `LoadVideo`: is the widget called `file`? Do uploaded mp4s appear in its list?
- [ ] `SaveVideo`: `format`/`codec` values (`auto`?), `filename_prefix`
- [ ] `CreateVideo`: input names `images`, `fps` (optional `audio`)
- [ ] `GetVideoComponents`: **is output slot 2 the fps?** (Assumed order: images, audio, fps.) Workflow node 43 uses `["11", 2]`.
- [ ] `CLIPVisionEncode`: does `crop` accept `none`?
- [ ] `ImageToMask`: `channel` = `red`

`app/main.py` patches workflow nodes by ID (10, 12, 15, 18, 5, 30, 40, 44). If you renumber nodes, update the app too.

## 3. aarch64 issues
- [ ] **torch CUDA:** see step 0.
- [ ] **onnxruntime-gpu:** there may be no aarch64 wheel. Try NVIDIA's Jetson/SBSA builds. If nothing works, fall back to CPU `onnxruntime` (DWPose gets slower) or use the torchscript bbox detector if controlnet_aux offers one. Record what you chose.
- [ ] **SAM2 CUDA extension:** if `pip install git+https://github.com/facebookresearch/sam2.git` fails to compile, reinstall with `SAM2_BUILD_CUDA=0`. The extension only does hole-filling post-processing, so masks still work.
- [ ] Note any other wheels that failed (decord, av, opencv) and how you fixed them.

## 4. Smoke runs (2 s clip, 480p)
Use a roughly 2 s, 832×480 clip (set `length` to 33 for recast/swap/restyle; keep 4n+1 frames). Run each workflow through the app (`uvicorn app.main:app --port 7860`) and also directly by POSTing the JSON to `/prompt`.

| Workflow | Wall time | Peak memory (`nvidia-smi` / `free -g`) | OK? | Notes |
|---|---|---|---|---|
| recast_wan22_animate | | | | |
| swap_wan_vace | | | | |
| restyle_wan_vace_depth | | | | |

Then run each once at the full 81 frames (77 for recast) and add those numbers to the table.

## 5. Mask quality (swap)
- [ ] Check `work/<sid>/mask_preview.png` and the output for halos or seams at the object edge. The mask passes through an mp4v file, so compression may blur its edges.
- [ ] Things to try if it looks bad: write the mask as a lossless PNG sequence or FFV1 instead, threshold after loading, or tune `GrowMask expand` (default 8) and `tapered_corners`. Record the best values in the presets.

## 6. Sampler tuning
The starting values are 20 steps, cfg 5.0, uni_pc/simple, shift 8.0, VACE strength 1.0. All untuned.
- [ ] Sweep steps (12/20/30), cfg (3/5/6) and shift (5/8) on one clip per mode. Pick the best quality/time tradeoff and update `presets/*.json` and the workflow defaults.
- [ ] Optional (open weights only): test a speed LoRA such as lightx2v. Note its license.

## 7. TODOs to own
- [ ] **Chunking for long clips (4–30 s):** for recast, loop over `WanAnimateToVideo` using the `video_frame_offset` and `continue_motion` outputs. For VACE, use overlapping windows and crossfade them. Do this in the app (several `/prompt` calls) or as a looped workflow.
- [ ] **Comfy Developer Platform integration (contest requirement):** the entry *must* be built on the Comfy API, Comfy Router and/or Comfy SDK. Router is out (open-weights rule). Deploy the three workflows through **Comfy API** and/or use the **Comfy SDK** in `app/`. Document this in the README, and include a hosted endpoint link if you can (the contest strongly encourages one).

## 8. Report and commit
Send back a short report covering:
1. Environment: torch/CUDA versions, ComfyUI commit, controlnet_aux commit.
2. Each workflow's pass/fail plus the filled-in table from §4.
3. Every node or widget fix, as old → new.
4. aarch64 workarounds.
5. Mask and sampler findings, plus the final defaults.
6. Status of the open TODOs (chunking, Comfy SDK/API).
7. Links to 2–3 sample outputs.

Commit the fixes on a branch (`spark-bringup`) and open a **PR against `main`** at https://github.com/SydFloyd/openjutsu. Don't push directly to main.
