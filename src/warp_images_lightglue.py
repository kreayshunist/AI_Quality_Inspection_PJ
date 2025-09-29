import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

# LightGlue + feature extractor
from lightglue import LightGlue, SuperPoint  # you can swap to DISK if desired
from lightglue.utils import load_image, rbd


torch.set_grad_enabled(False)


def warp_directory_with_lightglue(
    input_dir: Path,
    output_dir: Path,
    reference: Path | None = None,
    device: str | None = None,
    max_num_keypoints: int = 1024,
    feature: str = "superpoint",
    save_reference_copy: bool = True,
):
    """Warp all images in `input_dir` to the coordinate frame of `reference`
    using LightGlue matches and a RANSAC homography. Saves results to `output_dir`.

    - Chooses the first image in `input_dir` as reference if not provided.
    - Uses `SuperPoint` by default. Set `feature="disk"` to use DISK (requires import/update).
    - Writes warped images with the same filenames to `output_dir`.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dir = input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if len(files) == 0:
        print(f"[warp] No images found in {input_dir}")
        return

    # Pick reference
    if reference is None:
        reference = files[0]
    else:
        reference = Path(reference)
        if not reference.exists():
            raise FileNotFoundError(f"Reference image not found: {reference}")

    # Load extractor and matcher
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    #try:
    #    matcher.compile(mode="reduce-overhead")
    #except Exception:
        # compile may not be available on some torch builds; safe to skip
    #    pass
    print(f"[warp] Loaded LightGlue + {feature} on {device}")

    # Load reference image
    ref_cv = cv2.imread(str(reference), cv2.IMREAD_COLOR)
    if ref_cv is None:
        raise RuntimeError(f"Failed to read reference image: {reference}")

    # Optionally save reference image as-is into output directory
    if save_reference_copy:
        cv2.imwrite(str((output_dir / Path(reference).name)), ref_cv)

    ref_img = load_image(str(reference))  # CHW RGB float in [0,1]
    print("[warp] Using reference image:", reference)
    h0, w0 = ref_img.shape[-2:]
    feats0 = extractor.extract(ref_img.to(device))

    # Iterate other images (including reference again is harmless but we already saved it)
    total = len(files) - 1 if reference in files else len(files)
    done = 0
    for path in files:
        if path == reference:
            print(f"[warp][skip] Skipping reference image {path}")
            continue

        image1 = load_image(path)
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0_rb, feats1_rb, matches01_rb = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        kpts0, kpts1, matches = feats0_rb["keypoints"], feats1_rb["keypoints"], matches01_rb["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        h, status = cv2.findHomography(m_kpts1.cpu().numpy(), m_kpts0.cpu().numpy(), cv2.RANSAC)
        print(h)

        # Prepare current image for warping (OpenCV expects BGR uint8)
        img_rgb = (
            image1.permute(1, 2, 0).cpu().numpy() * 255.0
        ).clip(0, 255).astype(np.uint8)
        img_bgr = np.ascontiguousarray(img_rgb[..., ::-1])

        # Warp current image into reference frame size (w0, h0)
        warped_bgr = cv2.warpPerspective(img_bgr, h, (w0, h0))

        # Keep consistency with original code: flip back to RGB before saving
        aligned_rgb = warped_bgr[..., ::-1].copy()

        out_path = output_dir / path.name
        ok = cv2.imwrite(str(out_path), aligned_rgb)
        done += 1
        if ok:
            print(f"[warp][{done}/{total}] Wrote {out_path}")
        else:
            print(f"[warp][fail] Could not write {out_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Warp a directory of images using LightGlue")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/patchcore_data/train/good",
        help="Directory with input images to warp",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/patchcore_data_warped/train/good",
        help="Directory where warped images are saved",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional explicit path to reference image (defaults to first in input_dir)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on: cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=1024,
        help="Max keypoints for the feature extractor",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="superpoint",
        choices=["superpoint"],
        help="Feature extractor to use (currently superpoint)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    warp_directory_with_lightglue(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        reference=args.reference,
        device=args.device,
        max_num_keypoints=args.max_keypoints,
        feature=args.feature,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
