# SPACE : a Lightweight AI based Vision System for Smart Parking Effeciency 

Smart Parking Application for Circulation Efficiency (SPACE) is a parking lot automatic occupancy detection using the TensorFlow Object Detection API, targeting deployment on the **Alif Ensemble E7** edge device.

Datasets: [PKLot](https://public.roboflow.com/object-detection/pklot) (public, 12k images) and custom dataset, 500 images collected at Telit Cinterion, Trieste.

## Setup

Clone with submodule:

```powershell
git clone --recurse-submodules https://github.com/poorni666/Automated-Vision-system-for-Smart-parking-efficiency.git
cd SPACE

# if you forgot --recurse-submodules:
git submodule update --init --recursive
```

Create a virtual environment and install dependencies (Python 3.7, Windows):

```powershell
py -3.7 -m venv .venv
.venv\Scripts\Activate.ps1
python setup_local.py
```

`setup_local.py` handles everything: installs pinned requirements, downloads `protoc.exe` v3.19.6, compiles TF OD API protobufs, installs the `object_detection` package, and sets `PYTHONPATH`.

Verify:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"   # 2.11.0
python -c "from object_detection.utils import label_map_util; print('OD API OK')"
```

> **Docker alternative:** `docker compose up --build` (GPU) or `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build` (CPU). Jupyter at `localhost:8888`, TensorBoard at `localhost:6006`.

### Common setup issues

**`AlreadyExistsError: File system for az already registered`** — `tensorflow-io` registers the Azure filesystem plugin twice on Windows with TF 2.11. Fix:

```powershell
pip uninstall tensorflow-io tensorflow-io-gcs-filesystem -y
```

If it reappears after reinstalling, pin it: `pip install tensorflow-io==0.23.1`.

**`ModuleNotFoundError: No module named 'object_detection'`** — Re-run `python setup_local.py`, or add to your venv's `Activate.ps1`:

```powershell
$env:PYTHONPATH = "d:\SPACE\tensorflow_models\research;d:\SPACE\tensorflow_models\research\slim"
```

---

## Data

> **Important:** Datasets are not included. Download and place them as described below, then generate TFRecords.

**PKLot** — download the object detection version (TFRecord format, 320×320) from Roboflow:

```
datasets/pklot/
  train/  valid/  test/  test_images/  label_map.pbtxt
```

**TelitLot** — custom dataset collected at the Telit Cinterion facility (Trieste). Annotations in Pascal VOC XML format are in `datasets/custom/annotations/`. Raw images not included — contact the author.

```
datasets/custom/
  train/  valid/  test/  label_map.pbtxt
```

Generate TFRecords from annotations:

```powershell
python scripts/dataset_prep/convert_to_tfrecord.py --dataset pklot
python scripts/dataset_prep/convert_to_tfrecord.py --dataset custom
```

All paths and split configs live in `scripts/experiment_configs.py`.

---

## Pipeline Architecture

```
Images (320×320) ──► MobileNetV2 Backbone ──► FPN Neck ──► SSD Head ──► NMS ──► Predictions
                      (feature extraction)    (multi-scale)  (anchors)           (boxes + labels)
                            │
                     Pretrained on COCO17
                            │
                    ┌───────┴────────┐
               Fine-Tuning      Frozen Backbone
              (all layers)     (head only, E3/E6)
```

```
PKLot (12k imgs) ──► Train ──► Evaluate on PKLot    (E1, E3 — in-domain)
                 └──────────► Evaluate on TelitLot   (E2, E4 — cross-domain)

TelitLot (500 imgs) ──► Train ──► Evaluate on TelitLot  (E5, E6 — target-domain)

Best model (E6) ──► Export SavedModel ──► TFLite INT8 ──► Alif Ensemble E7
```

### Components

- **Backbone — MobileNetV2**: lightweight feature extractor using depthwise separable convolutions and inverted residual blocks. Pretrained on COCO 2017.
- **Neck — FPN Lite**: Feature Pyramid Network combines multi-scale feature maps so the model handles both nearby (larger) and distant (smaller) parking spaces in the same image.
- **Head — SSD**: single-pass prediction of bounding boxes and class scores over predefined anchors. No region proposals — fast enough for edge inference.
- **Post-processing — NMS**: removes duplicate overlapping detections per space.
- **Transfer learning strategies**: fine-tuning updates all weights; frozen backbone freezes `FeatureExtractor` and trains only the detection head — fewer trainable parameters, faster convergence, less data needed.

### Code Structure

```
configs/
  base_pipeline.config         # shared base — all experiments extend this
  experiments/
    pipeline_E1.config         # fine-tuning on PKLot
    pipeline_E3.config         # frozen backbone on PKLot
  config.py                    # path constants + config helpers

scripts/
  experiment_configs.py        # all experiment paths + hyperparameters (edit here)
  patch_config.py              # patches pipeline.config fields at runtime
  training_loop.py             # training orchestration
  evaluate_protocol.py         # PR curve, F1 threshold sweep, confusion matrices
  setup_local.py               # automated setup

notebooks/
  03_training_pklot.ipynb      # E1 + E3
  04_training_custom.ipynb     # E5 + E6
  05_training_general.ipynb    # general-purpose configurable training (edit via experiment_configs.py)
  05_evaluation.ipynb          # in-domain evaluation
  06_cross_dataset_eval.ipynb  # cross-domain evaluation
  08_inference_demo.ipynb      # visual demo + speed benchmark
```

---

## Usage

### Configure an experiment

All paths and hyperparameters are defined in `scripts/experiment_configs.py`. To change batch size, number of steps, learning rate, or toggle backbone freezing — edit it there. `patch_config.py` writes those changes into the pipeline config at runtime, so you never need to edit a `.config` file directly.

To switch between fine-tuning and frozen backbone in the pipeline config:

```protobuf
train_config {
  fine_tune_checkpoint: "models/pretrained/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0"
  fine_tune_checkpoint_type: "detection"

  # frozen backbone: add this line. Fine-tuning: remove it.
  freeze_variables: "FeatureExtractor"
}
```

### Run training

```powershell
# via notebook:
jupyter notebook notebooks/05_training_general.ipynb

# or directly:
python tensorflow_models/research/object_detection/model_main_tf2.py \
  --pipeline_config_path=configs/experiments/pipeline_E1.config \
  --model_dir=models/checkpoints/M1_pklot_fine_tune \
  --num_train_steps=50000 \
  --checkpoint_every_n=5000 \
  --alsologtostderr
```

Monitor with TensorBoard:

```powershell
tensorboard --logdir=models/checkpoints/
```

### Evaluate

```powershell
python scripts/evaluate_protocol.py \
  --pipeline_config=configs/experiments/pipeline_E1.config \
  --checkpoint_dir=models/checkpoints/M1_pklot_fine_tune \
  --eval_record=datasets/pklot/test/test.tfrecord \
  --label_map=datasets/pklot/label_map.pbtxt
```

Outputs: mAP@0.5, PR curve, optimal confidence threshold (max F1), existence + class confusion matrices.

### Export to TFLite

See `models/tflite/loadmodel.ipynb` for the INT8 quantization workflow using representative dataset calibration, and `notebooks/08_inference_demo.ipynb` for inference benchmarking across model formats.

---

## Experiments & Results

Six experiments across three evaluation scenarios. All use SSD MobileNetV2 FPNLite 320×320 initialized from COCO17 pretrained weights, Momentum optimizer, cosine decay LR schedule, and 50k training steps.

| Exp | Model | Train | Strategy | Test | mAP@0.5 | Threshold | F1 |
|-----|-------|-------|----------|------|:-------:|:---------:|:--:|
| E1 | M1 | PKLot | Fine-Tuning | PKLot | **0.92** | 0.27 | 0.84 |
| E2 | M1 | PKLot | Fine-Tuning | TelitLot | 0.00 | 0.21 | 0.14 |
| E3 | M2 | PKLot | Frozen Backbone | PKLot | **0.71** | 0.29 | 0.65 |
| E4 | M2 | PKLot | Frozen Backbone | TelitLot | 0.02 | 0.29 | 0.12 |
| E5 | M3 | TelitLot | Fine-Tuning | TelitLot | 0.00 | — | — |
| E6 | M4 | TelitLot | Frozen Backbone | TelitLot | **0.72** | 0.48 | 0.79 |

**E1/E3** — in-domain baselines on PKLot. Fine-tuning reaches mAP 0.92; frozen backbone drops to 0.71 because the fixed backbone misses some localisation, though classification of detected spaces stays accurate.

**E2/E4** — zero-shot cross-domain transfer to TelitLot. Both models fail (mAP ≈ 0). Domain shift from viewpoint, lighting, and scene geometry is too large for zero-shot transfer.

**E5/E6** — training directly on 500 TelitLot images. Fine-tuning (E5) fails — all-layer updates on very limited data disrupt pretrained features without enough target data to guide them. Frozen backbone (E6) achieves mAP 0.72, converging at ~10k steps. **E6 is the recommended deployment configuration.**

### Model compression

| Format | Precision | Size | Compression |
|--------|-----------|------|:-----------:|
| SavedModel | FP32 | 15.8 MB | 1× |
| TFLite INT8 | INT8 | **3.31 MB** | **4.8×** |

The quantized model fits within the Alif Ensemble E7's 13.5 MB SRAM constraint. Parameter count (3.5M) is unchanged.

---

## Model Choices

**SSD MobileNetV2 FPNLite 320×320** — chosen for three reasons: (1) the Alif E7's Ethos-U55 NPU supports TFLite Micro with INT8 quantization, which rules out PyTorch-based models and heavyweight architectures like DETR; (2) the 13.5 MB SRAM budget required a model under ~4 MB after quantization; (3) SSD's single-pass inference is fast enough for real-time edge deployment where two-stage detectors (Faster R-CNN) are too slow.

**Frozen backbone** — preserving COCO-pretrained features prevents catastrophic forgetting when fine-tuning on only 500 images. The detection head has enough capacity to learn parking-specific localisation and occupancy classification on top of general visual features. This also makes training feasible on CPU (batch size 2, no GPU required), which matters for reproducibility.

**INT8 quantization** — post-training quantization with a representative calibration dataset. Reduces model size 4.8× with negligible accuracy loss, and maps directly onto the E7's NPU inference workflow.

---

## Known Limitations

- **Cross-domain generalization requires target data.** Model trained on PkLot and tested on TelitLot failed to generalize. Some target-domain data is required ,500 images was sufficient with frozen backbone, but not with fine-tuning.
- **Fine-tuning on small datasets is unreliable.** E5 and E2 collapse under strict mAP evaluation. All-layer updates disrupt pretrained features when the dataset is too small or too different from the source. Use frozen backbone for any dataset under ~1k images.
- **mAP 0.00 doesn't always mean no detections.** E2 and E5 report 0.00 under strict COCO IoU ≥ 0.5, but inference images show some visually plausible detections. The IoU threshold rejects imprecise boxes — confusion matrix analysis and visual inspection are necessary complements to mAP.
- **TelitLot acquisition conditions were not fully controlled.** Some images have low-light and window reflections not present in PKLot. This adds domain variation beyond viewpoint and scene layout.
- **E6 converges early.** Best checkpoint is at 10k steps; training beyond that doesn't improve and may slightly hurt. Full 50k runs were kept for consistency with other experiments.
- **No export notebook (07).** Export workflow currently lives in `models/tflite/loadmodel.ipynb` and `08_inference_demo.ipynb`.

---

## Citation

```bibtex
@mastersthesis{krishnasamy2026space,
  author = {Krishnasamy Karthikeyan, Poornima Devi},
  title  = {Lightweight AI-based Vision System for Smart Parking Application},
  school = {University of Trieste},
  year   = {2026},
  note   = {MSc in Data Science \& Scientific Computing}
}
```
