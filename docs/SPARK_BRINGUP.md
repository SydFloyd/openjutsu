# DGX Spark bring-up

Measured on 2026-10-17 UTC on GB10, aarch64, 128 GB unified memory. The ComfyUI endpoint is represented as `http://<SPARK_HOST>:8191`; substitute the hostname for your deployment. Deployment-specific details are retained in local operator notes outside version control.

## Environment and models

CUDA is detected and a BF16 CUDA matrix multiply passed. The existing ARM CUDA runtime was reused through an isolated dependency overlay; the base environment was not upgraded.

| Component | Tested version / commit | Import |
|---|---|---|
| Python | 3.12.3, aarch64 | Pass |
| PyTorch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 | CUDA works |
| CUDA runtime / NVIDIA driver | 13.0 / 580.178.04 | GB10, capability 12.1 |
| ComfyUI | 0.41.0 / `fc04da91b2ab12f99e4716371a7e4f98ac982597` | Pass |
| ComfyUI-Manager | 3.42 / `4d3bed4b569205124a52444caf25de1719fae0de` | Pass |
| WanVideoWrapper | 1.4.7 / `088128b224242e110d3906c6750e9a3a348a659b` | Pass, 143 classes |
| VideoHelperSuite | 1.7.9 / `4d907bee61e92c2e65af3bd6383a4e4d356126d1` | Pass, 40 classes |
| controlnet_aux | 1.1.6 / `a587a4c1594524c1d2de5cd1a86cd868ff448e1c` | Pass, 64 classes |
| segment-anything-2 | 1.0.2 / `0c35fff5f382803e2310103357b5e985f5437f32` | Pass, 6 classes |
| KJNodes | 1.5.2 / `17195fcfa8a05f601b766a0f09c9733284f47416` | Pass, 263 classes |
| Official SDK / API proxy | 0.5.0 / 0.1.6 | Asset upload, jobs and downloads tested |
| App SAM2 | Meta source `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Real CUDA mask propagation tested |

Full package details: [spark-environment.json](spark-environment.json). Pack import success does not imply that every optional node in each pack was executed. These three graphs use core Wan nodes and controlnet_aux; the app uses Meta's SAM2 library.

All nine installed files (83,010,511,102 bytes total) passed SHA-256 verification. Paths below are relative to `<COMFYUI_DIR>/`. Pinned sources, hashes, sizes and licenses are in [spark-models.json](spark-models.json).

| Model | Installed path |
|---|---|
| Animate 14B, BF16 | `models/diffusion_models/wan2.2_animate_14B_bf16.safetensors` |
| VACE 14B, FP16, combined backbone and VACE | `models/diffusion_models/wan2.1_vace_14B_fp16.safetensors` |
| Wan 2.1 VAE, BF16 | `models/vae/Wan2_1_VAE_bf16.safetensors` |
| UMT5 XXL, FP16 | `models/text_encoders/umt5_xxl_fp16.safetensors` |
| CLIP vision H, FP16 | `models/clip_vision/clip_vision_h.safetensors` |
| SAM 2.1 Hiera Large, FP16 | `models/sam2/sam2.1_hiera_large-fp16.safetensors` |
| YOLOX TorchScript | `models/annotator/hr16/yolox-onnx/yolox_l.torchscript.pt` |
| DWPose TorchScript | `models/annotator/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt` |
| DepthAnything V2 Small | `models/annotator/depth-anything/Depth-Anything-V2-Small/depth_anything_v2_vits.pth` |

The FP16 text encoder is native CLIPLoader format; the wrapper loader includes conversion for those keys. Wrapper GPU workflows were not benchmarked. The VAE serves both Wan models. SAM2 uses the existing safetensors checkpoint with a strict state-dict load; it needs no duplicate `.pt` download. DepthAnything Small was selected for its Apache-2.0 license.

## Workflow compatibility fixes

All original class names were present. All three graphs pass `scripts/validate_workflows.py` against the live catalog of 1,196 classes. Original node IDs are retained.

| Where | Old | New |
|---|---|---|
| All, CLIPLoader | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `umt5_xxl_fp16.safetensors` |
| All, VAELoader | `wan_2.1_vae.safetensors` | `Wan2_1_VAE_bf16.safetensors` |
| Recast, DWPreprocessor bbox | `yolox_l.onnx` | `yolox_l.torchscript.pt` |
| Recast, DWPreprocessor face rendering | `enable` in the body pose image | `disable`, matching the native body/hand control branch |
| Restyle, depth checkpoint | `depth_anything_v2_vitl.pth` | `depth_anything_v2_vits.pth` |
| All, SaveVideo format | `auto` | `mp4` |
| All, SaveVideo codec | flat `codec: auto` | Removed; MP4's dynamic codec defaults to automatic |
| Swap, control video | Original RGB video sent directly to VACE | Core EmptyImage + ImageCompositeMasked fill the expanded edit region with RGB 127 before conditioning |
| Swap, decoded mask | ImageToMask directly into GrowMask | ThresholdMask at 0.5 before GrowMask, keeping the edit mask binary through video decoding |
| App, prompt field | `prompt.value` resolved to the browser's `window.prompt` function | Explicit DOM lookup sends the user's text |
| App, result handling | Forwarded core history output shape | Official SDK downloads output node 44; local result route supports video playback |
| App, mask transport | Lossy MP4 | Binary grayscale FFV1 MKV |

Confirmed without substitutions: LoadVideo uses `file`; GetVideoComponents outputs images/audio/fps (fps slot 2); CreateVideo accepts images/fps; CLIPVisionEncode accepts `crop=none`; ImageToMask accepts `channel=red`; DWPreprocessor accepts the selected TorchScript pose estimator, resolution 512, and `scale_stick_for_xinsr_cn=disable`.

## ARM compatibility

| Dependency | Evidence and selected fallback |
|---|---|
| decord | No matching CPython 3.12 Linux ARM binary. Use PyAV/OpenCV and ffmpeg. |
| SageAttention / xformers | Optional wheels were discoverable, but GB10 kernel compatibility was not established. Neither installed; actual renders use PyTorch SDPA. |
| onnxruntime-gpu | A 1.31.0 aarch64 CUDA 13 wheel exists; it was not installed or GPU tested. CPU onnxruntime 1.31.0 is installed; both DWPose stages actually use CUDA TorchScript. |
| SAM2 CUDA extension | Build disabled with `SAM2_BUILD_CUDA=0`; optional extension post-processing disabled. Real binary masks still propagate. Added missing `iopath`. |
| PyAV / OpenCV / ffmpeg | PyAV 18.1.0 and OpenCV headless 5.0.0 work. imageio-ffmpeg supplies an aarch64 ffmpeg 7.0.2 binary. |
| Other optional nodes | Wan FantasyPortrait warns about missing `onnx`; unrelated optional MediaPipe preprocessors were not provisioned. Requested packs import cleanly. |

## Benchmark method

Every render uses open downloaded weights, 832×480, 16 fps, batch 1, BF16/FP16 models and PyTorch SDPA. A 33-frame clip is 2.0625 seconds. Initial smoke runs use 20 steps, CFG 5, shift 8, uni_pc/simple, seed 20261016. Later runs record the selected settings separately.

Direct runs POST API-format graphs to `/prompt`. Browser runs exercise file upload → optional SAM2 click/track → preset → app → official SDK → local API proxy → ComfyUI → downloaded MP4 → playing browser video. Full/chunk runs use the same app endpoints and SDK. Setup/mask time is separate from generation time.

Memory is sampled every two seconds: ComfyUI's process allocation from `nvidia-smi`, host `/proc/meminfo`, and `/system_stats`. These are observed peaks, not allocator high-water marks. Unified CUDA and host memory views overlap and must not be added. Comfy's dynamic allocator also allocates outside the ordinary PyTorch counters, so the table uses the process measurement.

| Mode | Path | Frames | Steps/CFG/shift | Seconds / clip | Peak Comfy GiB | Min. host available GiB | Execution |
|---|---|---:|---|---:|---:|---:|---|
| Restyle | Direct smoke | 33 | 20/5/8 | 391.9 | 47.90 | 66.41 | Pass |
| Swap | Direct smoke | 33 | 20/5/8 | 382.2 | 47.87 | 63.19 | Pass |
| Recast | Direct smoke | 33 | 20/5/8 | 330.8 | 47.60 | 62.66 | Pass |
| Recast | Direct smoke (final) | 33 | 20/1/5 | 182.3 | 47.57 | 65.84 | Pass |
| Swap | Direct smoke (final) | 33 | 12/3/5 | 238.8 | 48.21 | 65.65 | Pass |
| Recast | Browser + SDK smoke | 33 | 20/5/8 | 326.7 | 47.73 | 62.57 | Pass |
| Restyle | Browser + SDK smoke | 33 | 20/5/8 | 347.0 | 48.01 | 63.85 | Pass |
| Swap | Browser + SDK smoke | 33 | 20/5/8 | 393.6 | 48.21 | 63.61 | Pass |
| Recast | App + SDK full window | 77 | 20/1/5 | 472.8 | 47.57 | 63.14 | Pass |
| Restyle | App + SDK full window | 81 | 20/3/8 | 1056.1 | 53.06 | 58.61 | Pass |
| Swap | App + SDK full window | 81 | 12/3/5 | 696.1 | 53.37 | 60.05 | Pass |
| Recast | App + SDK two windows | 80 | 20/1/5 | 489.0 | 47.60 | 64.22 | Pass |
| Restyle | App + SDK two windows | 80 | 20/3/8 | 990.1 | 49.06 | 62.77 | Pass |
| Swap | App + SDK two windows | 80 | 12/3/5 | 673.3 | 49.50 | 62.37 | Pass |

Full and two-window runs use the selected defaults below. Two-window runs force a 45-frame capacity and produce exactly 80 frames (5 seconds), with final padding removed. Complete settings, setup times, child job IDs and sweep timings: [spark-benchmarks.json](spark-benchmarks.json).

## Mask findings

SAM2 was exercised through the browser with an actual foreground click. The 33-frame test mask decodes as FFV1 with values exactly `{0,255}`. Mean intersection-over-union against the procedural red ball is **0.9641**; foreground area remains 6,000–6,145 pixels. The blue sphere replacement preserves the room. No obvious compression halo was visible at the sampled object boundaries. `GrowMask expand=8`, `tapered_corners=true` is retained for this fixture; thin objects and other footage need their own check.

The 81-frame tracking check reached **0.9675 mean IoU** (minimum 0.9532), with binary FFV1 values throughout. A server-side diagnostic found decoded mask values 0/254; thresholding at 0.5 produces 0/255. The corrected control video has exactly RGB 127 at the object centre, zero remaining red object pixels, and zero RGB difference outside the erased region when compared against the same ComfyUI-decoded source frame.

## Sampler findings

The sweep compares steps 12/20/30, then CFG 3/5/6 at the selected step count, then shift 5/8 at the selected steps/CFG. This is a coordinate sweep, not a full Cartesian search. Each mode keeps its source, reference, prompt and seed fixed. Animate also tests CFG 1, matching the [upstream base configuration](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_animate_14B.py). No speed LoRA or quantized backbone is used.

All **19 distinct configurations** completed successfully (three existing baseline runs were reused). The original defaults were 20 steps / CFG 5 / shift 8 for every mode. The selected values are applied to all three workflow files and all 13 presets.

| Mode | Steps | CFG | Shift | Selected 33-frame seconds | Peak Comfy GiB | Visual comparisons |
|---|---:|---:|---:|---:|---:|---|
| Recast | 20 | 1 | 5 | 180.2 | 47.64 | [Steps](tuning/recast-steps.jpg), [CFG](tuning/recast-cfg.jpg), [shift](tuning/recast-shift.jpg) |
| Swap | 12 | 3 | 5 | 236.5 | 48.21 | [Steps](tuning/swap-steps.jpg), [CFG](tuning/swap-cfg.jpg), [shift](tuning/swap-shift.jpg) |
| Restyle | 20 | 3 | 8 | 385.1 | 48.21 | [Steps](tuning/restyle-steps.jpg), [CFG](tuning/restyle-cfg.jpg), [shift](tuning/restyle-shift.jpg) |

- Recast: 12 steps softened the background; 30 added little detail for 478.6 seconds. CFG 1 kept coherent matched-reference motion at roughly half the CFG 3/5/6 time. Shift 5 had slightly clearer first-frame detail. DWPose QA found exactly one person in every source frame.
- Swap: 12 steps preserved the replacement and room about as well as 20/30 on this sphere fixture. CFG 3 and shift 5 reduced initial faceting and dark rim contrast. Some surface variation remains; more steps did not resolve it.
- Restyle: 20 steps improved background definition over 12; 30 was not worth 571.1 seconds. CFG 3 gave cleaner window edges; shift 8 retained more definition. This fixed tuning fixture is reference-conditioned clay style; the app tests use the 8mm preset without a reference.

These are visual selections on one clip and seed per mode, not universal optima or a broad evaluation of all 13 styles. Timings are single runs; small differences between equivalent compute settings are not statistically meaningful. `uni_pc/simple`, VACE strength 1, and mask expansion 8 with tapered corners are retained.

The Recast sweep used the handoff's mixed face/body pose image. A subsequent full app test with the cartoon reference returned 77 frames but duplicated the character, so it is an execution pass and a **quality failure**. The final graph renders body/hands only, following the native template; final Recast checks are repeated after that correction. The graph has no pose retargeting or separate expression crop. [Wan's preprocessing guide](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md) recommends retargeting for dissimilar body proportions. Use similarly framed subjects with this app; arbitrary cross-proportion transfer remains a quality limitation.

The initial 81-frame Swap execution also failed the requested colour change: it kept the sphere red despite a correctly bound blue reference and a good mask. Inspection found a missing preprocessing step: [VACE expects missing regions to be grey in the control video](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md). The corrected graph erases the expanded edit region to RGB 127 before conditioning. The follow-up also describes the desired blue sphere without naming the old red object. The short-clip sampler sweep predates this correction; final Swap tests are repeated with the corrected input.

The corrected full Recast run produced **77 frames in 472.8 seconds**, retaining one character with the matched Sintel reference ([comparison](quality/recast-corrected-full.jpg)). A final 33-frame run with the body-only graph and a manually aligned cartoon reference completed in **182.3 seconds** and removed the duplicate character, but badly distorted the legs ([comparison](quality/recast-aligned-cartoon-limit.jpg)). That remains a **quality failure**. The exported Recast sample uses the successful matched-reference run. The [original duplicated-character output](quality/recast-original-duplicate.jpg) is retained for comparison; this app does not provide automatic pose retargeting.

The corrected full Swap run produced **81 frames in 696.1 seconds**. A fixture-specific colour check found a blue sphere in **all 81 frames**, with at least 8,009 blue pixels per frame inside the expected tracked region. Sampled frames retain the room and a single sphere, with slight faceting at its lower edge ([corrected output](quality/swap-corrected-full.jpg), [original red-object failure](quality/swap-original-red.jpg), [grey control input](quality/swap-grey-control.png)). This colour check verifies this known fixture, not arbitrary object identity.

All three two-window tests decoded to exactly **80 frames at 16 fps**. Recast retains one character across the continuation ([join](quality/recast-two-window-join.jpg)); its test source repeats a short excerpt and contains source loop transitions. Swap stays blue through **all 80 frames**, with no obvious double edge in the sampled overlap ([join](quality/swap-two-window-join.jpg)). Restyle retains the moving ball, but furniture and lighting drift across windows; the crossfade smooths that change without eliminating it ([join](quality/restyle-two-window-join.jpg)).

## Long clips and SDK integration

The app normalizes uploads to 16 fps and processes the full frame count. Recast defaults to 77-frame windows, carries the last five generated frames through `continue_motion`, and advances `video_frame_offset` accounting for that context. VACE defaults to 81-frame windows with nine-frame overlap and linear crossfades. The final window pads to 4n+1, then output assembly removes padding and duplicate context. Original audio is muxed back into the result.

CPU assembly was also exercised at the full 30-second limit: seven production-size windows produced exactly 480 frames at 832×480 and 16 fps for both join methods. A complete 30-second diffusion render was not benchmarked.

The app uses the [official Comfy SDK](https://docs.comfy.org/development/api-development/sdks), with the self-hosted [API proxy](https://github.com/Comfy-Org/comfy-api-proxy). Assets, submission, status and output retrieval use the SDK workflow surface. There are no Router calls or partner/API nodes. ComfyUI runs with `--offline` and exposes zero partner/API classes.

Final checks: **17 CPU tests passed** for window coverage, final padding, lossless slicing/continuation, frame assembly and audio restoration. The dependency overlay's `pip check` reports no broken requirements. Every benchmark output was decoded and checked for 832×480, 16 fps and the expected frame count. Actual child job histories and memory samples are saved on Spark under `<REPORTS_DIR>/handoff/`.

After all GPU jobs finished, the temporary FastAPI app and SDK proxy were stopped. ComfyUI was restarted with the final dependencies: all six requested packs imported again, all three graphs passed the [fresh schema check](spark-workflow-validation.json), and the catalog contains **1,196 classes and zero partner/API classes**. The server runs `--listen 0.0.0.0 --port 8191 --offline --use-pytorch-cross-attention --disable-xformers --preview-method none --cache-none`.

Both `/system_stats` and `/object_info` returned HTTP 200 after the restart. A separate physical LAN client was not available for verification. The complete final catalog is saved at `<REPORTS_DIR>/object_info.json` (also `<REPORTS_DIR>/handoff/object_info-final.json`), with final stats and startup logs beside it.

Absolute installation paths and operational commands are retained in local operator notes outside version control. Keep the app and proxy on the laptop or another CPU host for ongoing use. No persistent app/dashboard service was added to Spark, and no public endpoint was deployed.

## Samples and credits

- [Recast: 77 frames](samples/recast.mp4), using the [Sintel reference](samples/reference-sintel.png) and [full motion fixture](samples/source-sintel-full.mp4).
- [Swap: 81 frames](samples/swap.mp4), using the [blue sphere reference](samples/reference-product.png) and [full ball fixture](samples/source-ball-full.mp4).
- [Restyle: 81 frames](samples/restyle.mp4), using the same ball fixture and the 8mm preset without a reference.

[Sample settings and provenance](samples/manifest.json), [credits](samples/README.md). The manually aligned cartoon candidate is excluded from these selected outputs because it failed visual QA.

The red-ball clip and illustrated character/product references are original procedural fixtures. Recast motion tests use a resized, cropped and retimed excerpt from the **Sintel trailer**, with the full-window fixture repeating the short excerpt. Attribution: **© copyright Blender Foundation | durian.blender.org**. Source: [official trailer](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4), [CC BY 3.0 sharing terms](https://durian.blender.org/sharing/), [license](https://creativecommons.org/licenses/by/3.0/). Recast samples are modified derivatives and retain that attribution. These fixtures establish compatibility and relative sampler behavior; they are not a broad quality evaluation.
