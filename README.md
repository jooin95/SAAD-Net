# SAAD-Net: Scale-Adaptive Anomaly-aware Defect Network

Official implementation of **"SAAD-Net: Scale-Adaptive Anomaly-aware Defect Network with Contrastive Learning for Multi-Class Defect Classification"**.

SAAD-Net is a classification architecture for industrial defect inspection, built on a Swin Transformer V2-Tiny backbone with three defect-specific modules:

- **LAA (Lightweight Anomaly-Aware Attention)** — amplifies defect-relevant regions from feature statistics.
- **S-SAA (Simplified Scale-Adaptive Attention)** — per-image learnable scale weighting across parallel dilated branches.
- **CDL (Contrastive Defect Learning)** — supervised contrastive objective with progressive warmup.

It is evaluated on a proprietary PEMFC electrode dataset (7 classes) and the public GC10-DET steel surface defect benchmark (11 classes).

---

## Results

Mean ± standard deviation over independent training runs with identical hyperparameters and different random seeds.

| Dataset | Split | SAAD-Net | Swin V2-T backbone | Runs |
|---|---|---|---|---|
| PEMFC electrode (7 classes) | validation, 280 images | **75.14 ± 2.34%** | 69.50 ± 1.76% | 5 |
| GC10-DET (11 classes) | validation, 689 images | **87.38 ± 0.94%** | 82.87 ± 0.28% | 3 |
| GC10-DET (11 classes) | held-out test, 701 images | **84.35 ± 1.51%** | 80.70 ± 3.07% | 3 |

Notes.

- The PEMFC validation set is class-balanced by augmentation (40 images per class); the underlying source-image counts differ per class.
- The GC10-DET test split is used for neither training nor checkpoint selection: each run is evaluated on it with the checkpoint chosen on the validation split.
- Efficiency (single NVIDIA RTX A5000, 1024×1024 input): 30.3M parameters, 96.8 GMACs, 28.0 FPS, 427 MB peak memory.

---

## Requirements

- Python 3.10, CUDA 11.8
- torch 2.0
- timm
- numpy, scipy, scikit-learn
- opencv-python, pillow
- matplotlib
- tqdm
- fvcore (for efficiency measurement)

```bash
pip install -r transfor_class/requirements.txt
```

---

## Repository layout

| Path | Purpose |
|---|---|
| `transfor_class/models/defect_lock_v2_improved.py` | **SAAD-Net model** (LAA, S-SAA, CDL modules) |
| `transfor_class/train_unified.py` | Main training / evaluation entry point (distributed, multi-GPU) |
| `transfor_class/eval/`, `transfor_class/utils/` | Evaluation and supporting utilities |
| `transfor_class/run_gc10_benchmark.sh` | GC10-DET benchmark driver |
| `results/gc10/` | Training history and confusion matrix of the representative GC10-DET run |

> **Note on imports.** `train_unified.py` imports the model as `from defect_lock_v2_improved import ...`, so make sure `defect_lock_v2_improved.py` is importable from the working directory (run from `transfor_class/` with `models/` on the `PYTHONPATH`, or place `defect_lock_v2_improved.py` in that directory). The repository also contains earlier development variants (`defect_lock.py`, `defect_lock_v2.py`, `saad_net_origin.py`, `train_saad_v3.py`, and other `train_*.py` scripts) that are **not** used for the reported results.

> **Not included here.** The GC10-DET preprocessing script and the efficiency-measurement script are not part of this repository; they are available from the corresponding author upon reasonable request, together with the PEMFC preprocessing pipeline.

---

## Datasets

**PEMFC electrode dataset.** Proprietary, collected from a live fuel-cell manufacturing line; cannot be redistributed due to industrial confidentiality. Available from the corresponding author upon reasonable request.

**GC10-DET.** Publicly available at https://github.com/lvxiaoming2019/GC10-DET.

### GC10-DET preprocessing

Each original 2048×1000 image is split along the vertical centerline into two halves, each resized to 1024×1024. A half inherits every annotated box that places at least 30% of its area within it, and is labeled by the largest-area rule; a half with no remaining annotated defect is labeled **Good**, giving 11 classes. The 70/15/15 train/validation/test split is stratified and performed at the original-image level before bisection (seed 42), so both halves of an image stay in the same split. This yields **3,224 training, 689 validation and 701 test** images.

---

## Usage

### Training

```bash
# GC10-DET (11 classes), 4 GPUs, distributed
torchrun --nproc_per_node=4 train_unified.py \
    --data_dir ./gc10_cropped \
    --model defect_lock_v2_imp \
    --num_classes 11 \
    --img_size 1024 \
    --epochs 200 \
    --batch_size 4 \
    --lr 1e-4

# PEMFC (7 classes), 8 GPUs, distributed
torchrun --nproc_per_node=8 train_unified.py \
    --data_dir /path/to/pemfc_processed \
    --model defect_lock_v2_imp \
    --num_classes 7 \
    --img_size 1024 \
    --epochs 200 \
    --batch_size 8 \
    --lr 5e-5
```

Both datasets use AdamW, cosine annealing with a 5-epoch warmup, early stopping with patience 20, dropout 0.3, focal loss (γ = 2.0, label smoothing 0.1), contrastive temperature 0.07 and a 10-epoch CDL warmup.

The full SAAD-Net model corresponds to `--model defect_lock_v2_imp` with the default config. Baselines in the paper are selected via `--model` (e.g. `swin_v2_t`, `vit_base`, `efficientnet_b4`, `resnet50`).

---

## Reproducing the paper

- Backbone: Swin Transformer V2-Tiny pretrained on ImageNet-22K.
- Evaluation aggregates metrics across all GPUs (`dist.all_reduce`) and reports on the full validation split; the GC10-DET test split is then evaluated once with the same best-validation checkpoint, without re-selection.
- Random seed fixed (`seed=42`) for dataset splitting; training seeds differ across runs.
- Reported accuracies are means over independent runs: five runs for the configurations the main claims rest on (SAAD-Net, the Swin V2-T baseline, EfficientNet-B4, the removal-block variants and the preprocessing ablation) and three for the remaining PEMFC rows; three runs for every GC10-DET model.
- Per-class tables and confusion matrices in the paper are shown for a representative run rather than averaged: PEMFC 75.71% at epoch 17, GC10-DET 87.08% at epoch 172.

---

## Citation

```bibtex
@article{joo2026saadnet,
  title   = {SAAD-Net: Scale-Adaptive Anomaly-aware Defect Network with Contrastive Learning for Multi-Class Defect Classification},
  author  = {Joo, In and Kim, Sung-Hoon and Seo, Mintaek and Ryu, Ga-Ae and Yoo, Kwan-Hee},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## License

See `transfor_class/LICENSE`.
