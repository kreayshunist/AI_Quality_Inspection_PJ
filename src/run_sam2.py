import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from helper import zero_out_background, crop_background, update_epochs

SAM2_SHA = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def sh(cmd, cwd=None):
    print("[cmd]", " ".join(map(str, cmd)), f"(cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=cwd)


# Default Cuda
def check_device(requested: str) -> str:
    if requested != "cuda":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--use-cluster", type=int, default=0)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()


    ROOT = Path(__file__).resolve().parents[1]
    REPO = ROOT / "sam2_repo"
    SAM2_IMAGES = ROOT / "data" / "sam2_data" / "samples"
    MASKS_OUT = ROOT / "output" / "masks"
    ZERO_OUT = ROOT / "output" / "masked" / "zero_out"
    CROP_OUT = ROOT / "output" / "masked" / "crop"

    for p in [MASKS_OUT, ZERO_OUT, CROP_OUT]:
        p.mkdir(parents=True, exist_ok=True)

    final_device = check_device(args.device)
    print(f"Selected Device: {final_device}")


    # Clone SAM2.1
    if not (REPO / ".git").exists():
        sh(["git", "clone", "https://github.com/facebookresearch/sam2.git", "sam2_repo"], cwd=ROOT)
    sh(["git", "fetch", "--all"], cwd=REPO)
    sh(["git", "checkout", SAM2_SHA], cwd=REPO)
    sh([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], cwd=REPO)

    # Get: default checkpoints
    dl_script = REPO / "checkpoints" / "download_ckpts.sh"
    checkpoints_dir = REPO / "checkpoints"
    # Only download if there are no .pt checkpoint files present
    pt_files = list(checkpoints_dir.glob("*.pt"))
    if dl_script.exists():
        if len(pt_files) == 0:
            dl_script.chmod(0o755)
            sh(["bash", "download_ckpts.sh"], cwd=checkpoints_dir)
        else:
            print(f"[Info] Found {len(pt_files)} .pt file(s) in {checkpoints_dir}; skipping checkpoints download.")
    else:
        print("[Error] checkpoints/download_ckpts.sh not found; skipping checkpoints download.")

    # CP train.yaml -> sam2_repo/sam2/configs/train.yaml
    SRC_YAML = ROOT / "config" / "train.yaml"
    DST_YAML = REPO / "sam2" / "configs" / "train.yaml"
    DST_YAML.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_YAML, DST_YAML)
    print(f"Copied {SRC_YAML} -> {DST_YAML}")


    if args.epochs is not None:
        update_epochs(DST_YAML, DST_YAML, int(args.epochs))


    #Train
    accel = "cuda" if final_device == "cuda" else "cpu"
    num_gpus = args.num_gpus if accel == "cuda" else 1
    train_cmd = [
        sys.executable, "training/train.py",
        "-c", "configs/train.yaml",
        "--use-cluster", str(int(bool(args.use_cluster))),
        "--num-gpus", str(num_gpus),
    ]
    sh(train_cmd, cwd=REPO)


    tuned_checkpoint = REPO / "sam2_logs" / "configs" / "train.yaml" / "checkpoints" / "checkpoint.pt"


    #Inference
    try:
        import sam2  # noqa: F401
    except ModuleNotFoundError:
        sam2_repo_path = str(REPO.resolve())
        if sam2_repo_path not in sys.path:
            sys.path.insert(0, sam2_repo_path)
        import sam2  # noqa: F401

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor


    if final_device == "cuda":
        try:
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Build model w/ the tuned checkpoint
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2 = build_sam2(model_cfg, str(tuned_checkpoint), device=final_device)
    predictor = SAM2ImagePredictor(sam2)

    # Mask Prompts: box, point (based on img size: 1024 * 1024)
    input_box = np.array([44, 102, 1016, 814])  # (x_min, y_min, x_max, y_max)
    input_point = np.array([[523, 686]])
    input_label = np.array([1])  # foreground

    images = [f for f in os.listdir(SAM2_IMAGES) if f.lower().endswith((".jpg", ".png"))]


    autocast_cm = torch.autocast("cuda", dtype=torch.bfloat16) if final_device == "cuda" else contextlib.nullcontext()

    with autocast_cm:
        for name in sorted(images):
            img_path = SAM2_IMAGES / name
            img_rgb = np.array(Image.open(img_path).convert("RGB"))

            predictor.set_image(img_rgb)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
            )
            mask_bool = masks[0] > 0

            # Save: /output/masks
            mask_img = (mask_bool.astype(np.uint8) * 255)
            (MASKS_OUT / f"{Path(name).stem}_mask.png").parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_img).save(MASKS_OUT / f"{Path(name).stem}_mask.png")

            # Save: /output/masked/zero_out
            zero_img = zero_out_background(img_rgb, mask_bool)
            ZERO_OUT.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(zero_img).save(ZERO_OUT / name)

            # Save: /output/masked/crop
            crop_img = crop_background(img_rgb, mask_bool)
            CROP_OUT.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop_img).save(CROP_OUT / name)

    print(f"Saved Zero-out | Check: {ZERO_OUT}")
    print(f"Saved Crop | Check: {CROP_OUT}")


if __name__ == "__main__":
    main()
