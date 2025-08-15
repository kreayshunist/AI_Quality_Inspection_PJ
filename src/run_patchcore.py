import argparse, sys, subprocess
from pathlib import Path
import os, json
import pandas as pd
from helper import (
    merge_w_labels,
    compute_metrics,
    save_result_reports,
    roc_threshold,
    save_heatmaps
)


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
    TEST_NORMAL_DIR = ROOT / "data" / "patchcore_data" / "test" / "good"
    TEST_ABNORMAL_DIR = ROOT / "data" / "patchcore_data" / "test" / "bad"
    PREDICT_DIR = ROOT / "output" / "masked" / args.masked
    LABELS_CSV = ROOT / "data" / "patchcore_data" / "label.csv"
    IMAGE_SIZE = (256, 256)
    OUT_DIR = ROOT / "output"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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


    # 2) Custom Datamodule by anomalib
    img_preprocess = v2.Compose([v2.Resize(IMAGE_SIZE), v2.ToTensor()])
    dm = Folder(
        name="plastic_part",
        root=None,
        normal_dir=TRAIN_DIR,
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


    # Inference
    dataset = PredictDataset(path=PREDICT_DIR, image_size=IMAGE_SIZE)
    predictions = engine.predict(model=model, dataset=dataset)
    print(f"Predictions: {len(predictions)}")

    # predictions -> DataFrame 
    if not predictions or not (hasattr(predictions[0], "pred_score") and hasattr(predictions[0], "pred_label")):
        raise RuntimeError("predictions is empty")



    pred_rows = []
    for p in predictions:
        path = p.image_path[0] if isinstance(p.image_path, list) else p.image_path
        score = float(p.pred_score.flatten()[0]) # normalized pred_score by anomalib
        label = int(p.pred_label)    
        pred_rows.append({
            "filename": os.path.basename(str(path)),
            "pred_score": score,
            "predicted_label": label,
        })
    pred_df = pd.DataFrame(pred_rows) 


    # Map with GT labels
    labeled_df = merge_w_labels(pred_df, LABELS_CSV)  

    threshold_info = None

    # Case 1: ROC curve based threshold 
    if args.threshold is not None:
        print("Apply ROC Curve threshold")
        y_true = labeled_df["label"].astype(int).to_numpy()
        scores = labeled_df["pred_score"].astype(float).to_numpy()

        thr, info = roc_threshold(y_true, scores, max_fnr=float(args.threshold))
        threshold_info = {"strategy": "roc_youden_fnr", "threshold": thr, **info}

        pred_df["predicted_label"] = (pred_df["pred_score"] >= thr).astype(int)
        labeled_for_metrics = merge_w_labels(pred_df, LABELS_CSV)

    # Case 2: Adaptive F1 based threshold
    else:
        print("Apply Adaptive F1 threshold")
        threshold_info = {"strategy": "adaptive_f1"}
        labeled_for_metrics = labeled_df


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