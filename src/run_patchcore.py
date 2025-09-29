import argparse, sys, subprocess
from pathlib import Path
import os, json
import pandas as pd
import torch
from helper import (
    merge_w_labels,
    compute_metrics,
    save_result_reports,
    roc_threshold,
    save_heatmaps
)
from warp_images_lightglue import warp_directory_with_lightglue

torch.set_grad_enabled(False)

def sh(cmd, cwd=None):
    print("[cmd]", " ".join(map(str, cmd)), f"(cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Give a maxFNR(0~1) for a ROC based threshold"
    )
    ap.add_argument(
        "--masked",
        type=str,
        default="zero_out",
        help="zero_our or crop"
    )
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    REPO = ROOT / "anomalib"
    TRAIN_DIR = ROOT / "data" / "patchcore_data" / "train" / "good"
    TRAIN_DIR_WARPED = ROOT / "data" / "patchcore_data_warped" / "train" / "good"
    TEST_NORMAL_DIR = ROOT / "data" / "patchcore_data" / "test" / "good"
    TEST_ABNORMAL_DIR = ROOT / "data" / "patchcore_data" / "test" / "bad"
    TEST_NORMAL_DIR_WARPED = ROOT / "data" / "patchcore_data_warped" / "test" / "good"
    TEST_ABNORMAL_DIR_WARPED = ROOT / "data" / "patchcore_data_warped" / "test" / "bad"
    PREDICT_DIR = ROOT / "output" / "masked" / args.masked
    PREDICT_DIR_WARPED = ROOT / "output" / "masked_warped" / args.masked
    LABELS_CSV = ROOT / "data" / "patchcore_data" / "label.csv"
    IMAGE_SIZE = (256, 256)
    OUT_DIR = ROOT / "output"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR_WARPED.mkdir(parents=True, exist_ok=True)
    TEST_NORMAL_DIR_WARPED.mkdir(parents=True, exist_ok=True)
    TEST_ABNORMAL_DIR_WARPED.mkdir(parents=True, exist_ok=True)
    PREDICT_DIR_WARPED.mkdir(parents=True, exist_ok=True)

    # 0) Determine a stable reference image from training set
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    train_files = sorted([p for p in TRAIN_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if len(train_files) == 0:
        raise RuntimeError(f"No training images found in {TRAIN_DIR}")
    reference_image = train_files[0]
    print(f"[warp] Using reference image for alignment: {reference_image}")

    # 1) Warp training images into TRAIN_DIR_WARPED using LightGlue
    warp_directory_with_lightglue(
        input_dir=TRAIN_DIR,
        output_dir=TRAIN_DIR_WARPED,
        reference=reference_image,
    )

    # anomalib clone -> checkout -> install
    if not (REPO / ".git").exists():
        sh(["git", "clone", "https://github.com/open-edge-platform/anomalib.git", str(REPO.name)], cwd=ROOT)
    sh(["git", "checkout", "f6ec1c57363a9894ff57184a5bfb78efa8f3de1b"], cwd=REPO)
    sh([sys.executable, "-m", "pip", "install", "-e", ".[dev,cpu]"], cwd=REPO)


    try:
        import anomalib  
    except ModuleNotFoundError:
        src_path = str((REPO / "src").resolve())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        import anomalib  

    from anomalib.data import Folder, PredictDataset
    from anomalib.data.utils import TestSplitMode, ValSplitMode
    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from anomalib.post_processing import PostProcessor
    from torchvision.transforms import v2

    # 1.5) Warp the test (inference) images to the same reference
    warp_directory_with_lightglue(
        input_dir=TEST_NORMAL_DIR,
        output_dir=TEST_NORMAL_DIR_WARPED,
        reference=reference_image,
        save_reference_copy=False,
    )
    warp_directory_with_lightglue(
        input_dir=TEST_ABNORMAL_DIR,
        output_dir=TEST_ABNORMAL_DIR_WARPED,
        reference=reference_image,
        save_reference_copy=False,
    )
    # 2) Custom Datamodule by anomalib
    img_preprocess = v2.Compose([v2.Resize(IMAGE_SIZE), v2.ToTensor()])
    dm = Folder(
        name="plastic_part",
        root=None,
        normal_dir=TRAIN_DIR_WARPED,
        abnormal_dir=TEST_ABNORMAL_DIR,
        normal_test_dir=TEST_NORMAL_DIR,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.5,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=2,
        train_augmentations=img_preprocess,
        val_augmentations=img_preprocess,
        test_augmentations=img_preprocess,
    )
    dm.setup()

    # PostProcessor to find an optiimal threshold by F1AdaptiveThreshold by amonalib
    post_processor = PostProcessor()  

    # Train 
    model = Patchcore(coreset_sampling_ratio=1.0, post_processor=post_processor)
    engine = Engine()
    engine.train(datamodule=dm, model=model)


    # Inference on warped test images (good + bad)
    dataset_good = PredictDataset(path=TEST_NORMAL_DIR_WARPED, image_size=IMAGE_SIZE)
    dataset_bad = PredictDataset(path=TEST_ABNORMAL_DIR_WARPED, image_size=IMAGE_SIZE)
    predictions_good = engine.predict(model=model, dataset=dataset_good)
    predictions_bad = engine.predict(model=model, dataset=dataset_bad)
    predictions = list(predictions_good) + list(predictions_bad)
    print(f"Predictions: {len(predictions)} (good={len(predictions_good)}, bad={len(predictions_bad)})")

    # predictions -> rows for saving and arrays for metrics
    if not predictions:
        raise RuntimeError("predictions is empty")

    pred_rows = []
    y_true_list = []
    scores_list = []

    # good (label=0)
    for p in predictions_good:
        path = p.image_path[0] if isinstance(p.image_path, list) else p.image_path
        score = float(p.pred_score.flatten()[0]) if hasattr(p, "pred_score") else 0.0
        pred_label = int(p.pred_label) if hasattr(p, "pred_label") else 0
        pred_rows.append({
            "filename": os.path.basename(str(path)),
            "pred_score": score,
            "predicted_label": pred_label,
        })
        y_true_list.append(0)
        scores_list.append(score)

    # bad (label=1)
    for p in predictions_bad:
        path = p.image_path[0] if isinstance(p.image_path, list) else p.image_path
        score = float(p.pred_score.flatten()[0]) if hasattr(p, "pred_score") else 0.0
        pred_label = int(p.pred_label) if hasattr(p, "pred_label") else 0
        pred_rows.append({
            "filename": os.path.basename(str(path)),
            "pred_score": score,
            "predicted_label": pred_label,
        })
        y_true_list.append(1)
        scores_list.append(score)

    # Thresholding strategy
    if args.threshold is not None:
        print("Apply ROC Curve threshold")
        thr, info = roc_threshold(y_true_list, scores_list, max_fnr=float(args.threshold))
        threshold_info = {"strategy": "roc_youden_fnr", "threshold": thr, **info}
        for row in pred_rows:
            row["predicted_label"] = int(float(row["pred_score"]) >= thr)
    else:
        print("Apply Adaptive F1 threshold (model post-processor)")
        threshold_info = {"strategy": "adaptive_f1"}

    # Build final DataFrames
    pred_df = pd.DataFrame(pred_rows)
    y_pred_list = [int(r["predicted_label"]) for r in pred_rows]
    labeled_for_metrics = pd.DataFrame({
        "label": y_true_list,
        "predicted_label": y_pred_list,
    })


    metrics = compute_metrics(labeled_for_metrics)


    # Generate: predictions.csv & metrics.json
    save_result_reports(
        data=pred_df,           
        out_dir=OUT_DIR,
        metrics=metrics,        
        threshold_info=threshold_info if threshold_info is not None else {"strategy": "adaptive_f1"},
    )

    # 12) Heatmaps (★ 전체)
    filename_to_label = {fn: int(lbl) for fn, lbl in zip(pred_df["filename"], pred_df["predicted_label"])}

    print("-----Saving heatmaps-----")

    save_heatmaps(predictions, OUT_DIR, filename_to_label)

    print("\n======== PatchCore finished | Check:", OUT_DIR.resolve(), "\n")


if __name__ == "__main__":
    main()
