# README

## Setup

```bash
# Create and activate conda environment
conda create --name myenv python=3.11.13
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

## Run
```bash
python3 run.py --device cuda
```
## Architecture
<img width="983" height="428" alt="Screenshot 2025-08-15 at 2 19 03 AM" src="https://github.com/user-attachments/assets/9bba14cc-3586-4535-a02d-1f7d0253c8fb" />


## Directory Structure

```bash
project_root/
│
├── data/
│   ├── sam2_data/
│   │   └── samples/              # Dataset for SAM2.1 fine-tuning (.jpg & .json pairs)
|   |
│   ├── patchcore_data/
│   │   ├── train/
│   │   │   └── good/              # Normal images for PatchCore training
│   │   ├── test/
│   │   │   ├── good/             
│   │   │   └── bad/             
│   │   └── label.csv              # GT labels with filenames
│   │
│   └── hawk_quality/              # Provided GT labels 
│
├── output/
│   ├── masked/                     # zero_out & crop
│   ├── heatmap/                    
│   ├── overlay/            
│   ├── predictions.csv              # pred_score, pred_label
│   └── metrics.json                 # acc, f1, precision, recall, threshold type
│
├── src/
│   ├── helper.py                   # Helper functions for training/inference
│   ├── run_sam2.py                
│   └── run_patchcore.py           
│
├── notebooks/                      # Colab/Jupyter implementations
├── sample_output/                  # Expected output after running the code
└── utils/                          
    └── weak_aug.py                  # Data augmentation

```


## Flags
| Flag            | Type                     | Description                                                                                                                                                                                                                        |
| --------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--device`      | `"cuda"` or `"cpu"`      | *(Optional, default: `cuda`)* Selects the device for running the code.                                                                                                                                                             |
| `--epochs`      | `int`                    | *(Optional, default: `30`)* Number of epochs for SAM2.1 fine-tuning.                                                                                                                                                               |
| `--use-cluster` | `int`                    | *(Optional, default: `0`)* Number of clusters for SAM2.1 training.                                                                                                                                                                 |
| `--num-gpus`    | `int`                    | *(Optional, default: `1`)* Number of GPUs for SAM2.1 training.                                                                                                                                                                     |
| `--threshold`    | `float` or `None`        | *(Optional)* Maximum false positive rate (FPR) in range `[0, 1]`. If set, PatchCore uses the ROC curve threshold where FPR ≤ `maxFPR` (max Youden index). Without this flag, F1 adaptive threshold by anomalib is used by default. |
| `--masked`      | `"zero_out"` or `"crop"` | *(Optional, default: `zero_out`)* Applies either zero-out or crop background masking for anomaly detection with PatchCore.                                                                                                         |


## Notifications
- Developed for CUDA environments.

- PatchCore uses data from /output/masked for inference.
Clean this directory before running with different flags to avoid mixing results.

## Data Prerequisite
`data/sam2_data/samples`

- Required for SAM2.1 fine-tuning.

- Preprocessed using utils/coco_to_sam2.py to generate .jpg & .json pairs.

`data/patchcore_data`

- train: Normal images for training.

- test: Normal and anomalous images for testing.

  To train & test on unseen data, both train and test datasets are augmented using `utils/weak_aug.py`.

`data/patchcore_data/label.csv`

- Contains ground truth labels with corresponding filenames.




