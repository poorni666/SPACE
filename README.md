# S.P.A.C.E - Smart Parking Application for Circulation Efficiency with TF OD API
> Parking lot occupancy detection (PKLot + custom dataset) using the TensorFlow Object Detection API.
<p align="center">
  <img src="assests/parkingreadme.png" alt="Smart parking application illustration" width="900">
</p>
---

## Table of Contents
- [Project Overview](#project-overview)
- [Repo Structure](#repo-structure)
- [Quick Start (Docker)](#quick-start-docker)
- [Manual Setup (no Docker)](#manual-setup-no-docker)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation & Export](#evaluation--export)
- [Results](#results)

---

## Project Overview

This thesis investigates object detection for parking lot occupancy classification using:
- **PKLot** — a public benchmark dataset of parking lot images
- **Custom dataset** — collected and annotated manually

Models are trained using the **TensorFlow Object Detection API** and exported in multiple formats (SavedModel, TFLite, ONNX) for deployment analysis.

---

## Repo Structure

```
thesis-object-detection/
├── datasets/               # Data (raw images gitignored — see Datasets section)
│   ├── pklot/
│   └── custom/
├── notebooks/              # Numbered Jupyter notebooks (run in order)
├── scripts/
│   ├── dataset_prep/       # TFRecord generation, label maps, splits
│   ├── setup/              # Environment setup scripts
│   └── export/             # Model export scripts
├── configs/                # TF OD API pipeline.config files
├── models/                 # Trained model outputs (gitignored — see below)
├── results/                # Metrics and visualization outputs
├── tensorflow_models/      # Git submodule — tensorflow/models repo
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start (Docker)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with GPU support if using NVIDIA)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPU only)
- Git

### 1. Clone the repo (with submodule)
```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/thesis-object-detection.git
cd thesis-object-detection

# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

### 2. Pin the submodule to a stable commit
```bash
cd tensorflow_models
git checkout v2.11.0   # stable tag for TF 2.11
cd ..
git add tensorflow_models
git commit -m "pin tensorflow/models to v2.11.0"
```

### 3. Build and start
```bash
# GPU (recommended)
docker compose up --build

# CPU only — edit docker-compose.yml first:
# comment out the `deploy: resources:` section, then:
docker compose up --build
```

### 4. Open Jupyter
Visit → **http://localhost:8888**

TensorBoard → **http://localhost:6006**

---

## Manual Setup — Docker (recommended for others)
docker-compose up

## Setup — Local venv (faster iteration)
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements-dev.txt

> ⚠️ Requires Python 3.7 exactly. Use pyenv or conda to manage this.

```bash
# Create environment
conda create -n thesis python=3.7 -y
conda activate thesis

# Install dependencies
pip install -r requirements.txt

# Install TF OD API from submodule
git submodule update --init --recursive
bash scripts/setup/install_tf_od_api.sh

# Set PYTHONPATH
export PYTHONPATH=$(pwd)/tensorflow_models:$(pwd)/tensorflow_models/research:$(pwd)/tensorflow_models/research/slim:$PYTHONPATH

# Start Jupyter
jupyter notebook --notebook-dir=notebooks
```

---

## Datasets

### PKLot
PKLot is not included in this repo due to size. Download it from:
- **Kaggle**: https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset
- **Original paper site**: https://web.inf.ufpr.br/vri/databases/parking-lot-database/

Place images at: `datasets/pklot/raw/`

### Custom Dataset
The custom dataset annotations are in `datasets/custom/annotations/` (Pascal VOC XML format).
Raw images are not included — contact the author or see the thesis for acquisition details.

### Generate TFRecords
After placing images in the correct folders:
```bash
python scripts/dataset_prep/convert_to_tfrecord.py --dataset pklot
python scripts/dataset_prep/convert_to_tfrecord.py --dataset custom
```

---

## Training

```bash
# From inside Docker (or with PYTHONPATH set):
python tensorflow_models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=configs/pklot/pipeline.config \
    --model_dir=models/pklot/checkpoint \
    --alsologtostderr
```

Monitor training: open TensorBoard at http://localhost:6006

---

## Evaluation & Export

```bash
# Evaluate
python tensorflow_models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=configs/pklot/pipeline.config \
    --model_dir=models/pklot/checkpoint \
    --checkpoint_dir=models/pklot/checkpoint \
    --alsologtostderr

# Export SavedModel
python scripts/export/export_savedmodel.py --dataset pklot

# Export TFLite
python scripts/export/export_tflite.py --dataset pklot

# Export ONNX
python scripts/export/export_onnx.py --dataset pklot
```

---

## Results

| Dataset | Model | mAP@0.5 | Inference (ms) |
|---------|-------|---------|----------------|
| PKLot   | SSD MobileNetV2 | — | — |
| Custom  | SSD MobileNetV2 | — | — |

> Results will be populated after training. See `results/metrics/` for full evaluation output.

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `dataset_exploration.ipynb` | EDA on PKLot and custom dataset |
| 02 | `data_preparation.ipynb` | Annotation parsing, TFRecord generation |
| 03 | `training_pklot.ipynb` | Training walkthrough — PKLot |
| 04 | `training_custom.ipynb` | Training walkthrough — Custom dataset |
| 05 | `evaluation.ipynb` | mAP, precision/recall, confusion matrix |
| 06 | `inference_demo.ipynb` | Live inference with visualized detections |

---

## Citation

If you use this work, please cite the thesis:
```
[Your Name], "[Thesis Title]", [University], [Year].
```
