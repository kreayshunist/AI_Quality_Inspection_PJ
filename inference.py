import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from src.helper import zero_out_background, crop_background, save_heatmaps


def main():
    ap = argparse.ArgumentParser(description="Run inference using pretrained SAM2 and Patchcore models.")
    ap.add_argument("--input-dir", required=True, help="Directory containing input images")
    ap.add_argument("--sam2-repo", default="sam2_repo", help="Path to cloned SAM2 repository")
    ap.add_argument("--sam2-checkpoint", required=True, help="Path to trained SAM2 checkpoint (.pt)")
    ap.add_argument("--patchcore-checkpoint", required=True, help="Path to trained Patchcore checkpoint (.ckpt)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--masked", default="zero_out", choices=["zero_out", "crop"],
                    help="Masking strategy applied before anomaly detection")
    ap.add_argument("--out-dir", default="output/inference", help="Directory to save results")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    masked_dir = out_dir / "masked"
    out_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    # ----- SAM2 preparation -----
    sam2_repo = Path(args.sam2_repo).resolve()
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2 = build_sam2(model_cfg, args.sam2_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2)

    # Hardcoded prompts as in training
    input_box = np.array([44, 102, 1016, 814])
    input_point = np.array([[523, 686]])
    input_label = np.array([1])

    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))]
    for name in sorted(images):
        img_path = input_dir / name
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(img_rgb)
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        mask_bool = masks[0] > 0
        if args.masked == "zero_out":
            proc = zero_out_background(img_rgb, mask_bool)
        else:
            proc = crop_background(img_rgb, mask_bool)
        Image.fromarray(proc).save(masked_dir / name)

    # ----- Patchcore inference -----
    try:
        import anomalib
    except ModuleNotFoundError:
        repo_path = Path("anomalib") / "src"
        if repo_path.exists() and str(repo_path.resolve()) not in sys.path:
            sys.path.insert(0, str(repo_path.resolve()))
        import anomalib  # noqa: F401

    from anomalib.data import PredictDataset
    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from anomalib.post_processing import PostProcessor
    from torchvision.transforms import v2
    import pandas as pd

    IMAGE_SIZE = (256, 256)
    _ = v2.Compose([v2.Resize(IMAGE_SIZE), v2.ToTensor()])  # for completeness
    dataset = PredictDataset(path=masked_dir, image_size=IMAGE_SIZE)

    model = Patchcore.load_from_checkpoint(args.patchcore_checkpoint)
    model.post_processor = PostProcessor()
    model.eval()

    engine = Engine()
    predictions = engine.predict(model=model, dataset=dataset)

    pred_rows = []
    for p in predictions:
        path = p.image_path[0] if isinstance(p.image_path, list) else p.image_path
        score = float(p.pred_score.flatten()[0]) if hasattr(p, "pred_score") else float("nan")
        label = int(p.pred_label) if hasattr(p, "pred_label") else 0
        pred_rows.append({
            "filename": os.path.basename(str(path)),
            "pred_score": score,
            "predicted_label": label,
        })
    pred_df = pd.DataFrame(pred_rows)
    pred_csv = out_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    filename_to_label = {fn: int(lbl) for fn, lbl in zip(pred_df["filename"], pred_df["predicted_label"])}
    save_heatmaps(predictions, out_dir, filename_to_label)

    print(f"Saved predictions to {pred_csv}")
    print(f"Saved overlays and heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
