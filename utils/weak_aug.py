import os, random, cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

random.seed(42)

SRC_DIR = "./data/patchcore_data/predict/left_FLIR_Blackfly-S-BFS-U3-200S6C_23524388_1_1746110166.jpg"

OUT_DIR = "./data/patchcore_data/test/good"
os.makedirs(OUT_DIR, exist_ok=True)

base = Image.open(SRC_DIR).convert("RGB")

# Aug Params
ROT_DEG = 3        # ±3
SHIFT_P = 0.03     # ±3% 
BRIGHT = 0.05      # ±5%
CONTR  = 0.05      # ±5%
BLUR_P = 0.3       # 30% 

def weak_aug(img):
    out = img

    # Rotation
    angle = random.uniform(-ROT_DEG, ROT_DEG)
    out = F.rotate(out, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)

    # Translate
    tx = int(SHIFT_P * out.width  * random.uniform(-1, 1))
    ty = int(SHIFT_P * out.height * random.uniform(-1, 1))
    M = np.float32([[1,0,tx],[0,1,ty]])
    out = Image.fromarray(cv2.warpAffine(np.array(out), M, (out.width, out.height),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0))

    # Brightness
    b = 1.0 + random.uniform(-BRIGHT, BRIGHT)
    c = 1.0 + random.uniform(-CONTR,  CONTR)
    out = F.adjust_brightness(out, b)
    out = F.adjust_contrast(out,  c)

    # Blur
    if random.random() < BLUR_P:
        out_np = cv2.GaussianBlur(np.array(out), (3,3), 0)
        out = Image.fromarray(out_np)

    return out

# Num to generate
num_aug = 15 

#Save: an original img
base.save(os.path.join(OUT_DIR, "test_000_orig.jpg"))

#Save: generated imgs
for i in range(1, num_aug+1):
    aug = weak_aug(base)
    aug.save(os.path.join(OUT_DIR, f"test_{i:03d}.jpg"))

print(f"Saved {len(os.listdir(OUT_DIR))} images in {OUT_DIR}")