#!/usr/bin/env bash
# Downloads open-weights models into a ComfyUI install. Usage: COMFYUI_DIR=~/ComfyUI ./scripts/download_models.sh [--fp8]
set -euo pipefail
C="${COMFYUI_DIR:-$HOME/ComfyUI}/models"
mkdir -p models/sam2
dl(){ # repo path dest
  local out="$3/$(basename "$2")"; [ -f "$out" ] && { echo "have $out"; return; }
  echo "-> $out"; curl -L --fail --retry 3 --continue-at - -o "$out.part" "https://huggingface.co/$1/resolve/main/$2" && mv "$out.part" "$out"; }
sam2(){ dl Kijai/sam2-safetensors sam2.1_hiera_large-fp16.safetensors models/sam2; }
if [[ "${1:-}" == "--sam2-only" ]]; then sam2; exit; fi
mkdir -p "$C"/{diffusion_models,text_encoders,vae,clip_vision}
W22=Comfy-Org/Wan_2.2_ComfyUI_Repackaged; W21=Comfy-Org/Wan_2.1_ComfyUI_repackaged
if [[ "${1:-}" == "--fp8" ]]; then
  dl $W22 split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors "$C/diffusion_models"
  echo "NOTE: edit workflows/recast_wan22_animate.json unet_name to the int8 file for 24GB GPUs"
else
  dl $W22 split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors "$C/diffusion_models"
fi
dl $W21 split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors "$C/diffusion_models"
dl $W21 split_files/text_encoders/umt5_xxl_fp16.safetensors "$C/text_encoders"
dl Kijai/WanVideo_comfy Wan2_1_VAE_bf16.safetensors "$C/vae"
dl $W21 split_files/clip_vision/clip_vision_h.safetensors "$C/clip_vision"
sam2
# DWPose (yolox_l.torchscript.pt, dw-ll_ucoco_384_bs5.torchscript.pt) and Depth-Anything-V2 Small weights
# are auto-downloaded by comfyui_controlnet_aux on first run.
echo done
