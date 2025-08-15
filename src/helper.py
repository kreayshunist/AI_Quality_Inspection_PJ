from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import re
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd
from PIL import Image
try:
    import cv2
except Exception:
    cv2 = None



# ================= Helpers for SAM2.1 tasks ===============


# Update num_epochs in /config/train.yaml
def update_epochs(src_path: Path, dst_path: Path, epochs: int):

    text = src_path.read_text(encoding="utf-8")

    # num_epochs 라인만 교체
    new_text = re.sub(
        r'(?m)^(\s*num_epochs\s*:\s*)(\d+)(\s*.*)$',
        rf'\g<1>{epochs}\g<3>',
        text
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(new_text, encoding="utf-8")
    print(f"[ok] epochs overridden to {epochs} in {dst_path}")

# Output Dir: output/masked/zero_out
def zero_out_background(img: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[~mask_bool] = 0
    return out


# Output Dir: output/masked/crop
def crop_background(img: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return img[y0:y1+1, x0:x1+1]




# ================ Helpers for PatchCore tasks training  ==============

# Find optimal threshold using ROC curve
def roc_threshold(y_true, scores, max_fnr=0.0):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    # Single class fallback
    if np.unique(y_true).size < 2:
        thr = float(np.median(scores))
        return thr, {"reason": "single_class_fallback", "tpr": None, "fpr": None}

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]

    # No valid threshold candidates
    if thresholds.size == 0:
        thr = float(np.median(scores))
        return thr, {"reason": "no_finite_thresholds", "tpr": None, "fpr": None}

    youden = tpr - fpr
    min_recall = 1.0 - float(max_fnr)

    mask = (tpr >= min_recall)
    if np.any(mask):
        idxs = np.where(mask)[0]
        best_local = int(np.nanargmax(youden[idxs]))
        best_idx = idxs[best_local]
        reason = "youden_max_with_fnr_constraint"
    else:
        # No candidates left after FNR filter
        best_idx = int(np.nanargmax(tpr))
        reason = "fallback_max_recall"

    thr = float(thresholds[best_idx])
    return thr, {"reason": reason, "tpr": float(tpr[best_idx]), "fpr": float(fpr[best_idx])}


# 1. Load GT labels from data/patchcore_data/label.csv  
# 2. Match GT labels with images
def merge_w_labels(pred_df: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    gt_df = pd.read_csv(labels_csv)
    assert {"filename", "label"} <= set(gt_df.columns), "label.csv should have columns named filename & label"
    data = pred_df.merge(gt_df, on="filename", how="inner")
    if len(data) == 0:
        raise RuntimeError("No matching filename")
    return data



def compute_metrics(data_df: pd.DataFrame) -> Dict[str, float]:
    y_true = data_df["label"].astype(int).values
    y_pred = data_df["predicted_label"].astype(int).values
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        n_samples=int(len(data_df)),
    )




# ======== For heatmap & overlay ======
def tensor_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# Normalize heatmap to [0, 1]
def normalize_heatmap(h):
    h = np.asarray(h, dtype=np.float32)
    if h.ndim >= 3:
        h = np.squeeze(h)
    h_min, h_max = float(h.min()), float(h.max())
    return (h - h_min) / (h_max - h_min + 1e-8)



# Save overlay & heatmap
# Overlay Dir: output/heatmap_overlay
# Heatmap Dir: output/heatmap
def save_one_heatmap(
    init_path,
    heatmap_arr: np.ndarray,
    overlay_path: str | Path,
    heatmap_path: str | Path,
    score: Optional[float] = None,
    predicted_label: Optional[int] = None,
):
    if cv2 is None:
        raise RuntimeError("OpenCV has not installed")

    img_bgr = cv2.imread(str(init_path))
    # RGB->BGR
    if img_bgr is None:
        img_rgb = np.array(Image.open(init_path).convert("RGB"))
        img_bgr = img_rgb[..., ::-1].copy()  

    H, W = img_bgr.shape[:2]
    hm_resized = cv2.resize(heatmap_arr, (W, H), interpolation=cv2.INTER_LINEAR)
    hm_uint8 = (hm_resized * 255.0).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)

    # Annotate score & predicted label on overlay
    txt_parts = []
    if score is not None:
        txt_parts.append(f"Score={float(score):.4f}")
    if predicted_label is not None:
        txt_parts.append(f"Pred={int(predicted_label)}")
    if txt_parts:
        cv2.putText(overlay, " | ".join(txt_parts), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # save
    overlay_path = Path(overlay_path)
    heatmap_path = Path(heatmap_path)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(heatmap_path), hm_color)




def save_heatmaps(predictions, out_dir: Path | str, filename_to_label: Dict[str, int]) -> None:
    out_dir = Path(out_dir)
    heat_dir = out_dir / "heatmap"
    overlay_dir = out_dir / "heatmap_overlay"
    heat_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    total = len(predictions)
    for i, p in enumerate(predictions, 1):
        path = p.image_path[0] if isinstance(p.image_path, list) else p.image_path
        amap = getattr(p, "anomaly_map", None)
        if amap is None:
            continue

        h = tensor_to_numpy(amap)
        if h.ndim == 3:
            h = h[0]
        h = normalize_heatmap(h)

        fname = os.path.basename(str(path))
        stem, _ = os.path.splitext(fname)
        overlay_path = overlay_dir / f"{stem}_overlay.jpg"
        heatmap_path = heat_dir / f"{stem}_heatmap.jpg"

        score = float(p.pred_score.flatten()[0]) if hasattr(p, "pred_score") else None
        plabel = filename_to_label.get(fname)

        save_one_heatmap(path, h, overlay_path, heatmap_path, score=score, predicted_label=plabel)
        saved += 2
        if i % 50 == 0 or i == total:
            print(f"  - {i}/{total} processed")

    print(f"Saved {saved} images to:\n  - {overlay_dir}\n  - {heat_dir}")





# Generate & save predictions.csv & metrics.json
def save_result_reports(
    data: pd.DataFrame,
    out_dir: Path | str,
    metrics: Dict[str, float],
    threshold_info: Dict | float | None = None,
) -> Tuple[Path, Path]:
    """predictions.csv, metrics.json 저장."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = out_dir / "predictions.csv"
    data[["filename", "pred_score", "predicted_label"]].to_csv(pred_csv, index=False)

    metrics_json = out_dir / "metrics.json"
    payload = dict(metrics)
    if threshold_info is not None:
        payload["threshold"] = threshold_info
    with open(metrics_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", pred_csv)
    print("Saved:", metrics_json)
    return pred_csv, metrics_json