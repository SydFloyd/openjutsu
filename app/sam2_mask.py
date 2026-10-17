"""Click-to-mask with SAM 2.1 video predictor (facebookresearch/sam2). Writes a white-on-black mask video."""
import os, tempfile, pathlib
import cv2, numpy as np, torch

CKPT = os.environ.get("SAM2_CKPT", "models/sam2/sam2.1_hiera_large.pt")
CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
_pred = None

def _predictor():
    global _pred
    if _pred is None:
        from sam2.build_sam import build_sam2_video_predictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _pred = build_sam2_video_predictor(CFG, CKPT, device=dev)
    return _pred

def propagate_mask(video, points, out_mp4, preview_png):
    tmp = pathlib.Path(tempfile.mkdtemp())
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 16; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), f); i += 1
    cap.release()
    pred = _predictor()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        state = pred.init_state(video_path=str(tmp))
        pts = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        lbl = np.array([p[2] if len(p) > 2 else 1 for p in points], dtype=np.int32)
        pred.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=pts, labels=lbl)
        masks = {}
        for fi, _, logits in pred.propagate_in_video(state):
            masks[fi] = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
    h, w = masks[0].shape
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fi in sorted(masks): vw.write(cv2.cvtColor(masks[fi], cv2.COLOR_GRAY2BGR))
    vw.release()
    f0 = cv2.imread(str(tmp / "00000.jpg")); ov = f0.copy(); ov[masks[0] > 0] = (0, 0, 255)
    cv2.imwrite(str(preview_png), cv2.addWeighted(f0, 0.5, ov, 0.5, 0))
