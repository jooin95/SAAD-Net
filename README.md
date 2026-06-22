# SAAD-Net: Scale-Adaptive Anomaly-aware Defect Network

Official implementation of **"SAAD-Net: Scale-Adaptive Anomaly-aware Defect Network with Contrastive Learning for Multi-Class Defect Classification"** (Scientific Reports, 2026).

SAAD-Net is a classification architecture for industrial defect inspection, built on a Swin Transformer V2-Tiny backbone with three defect-specific modules:

- **LAA (Lightweight Anomaly-Aware Attention)** — amplifies defect-relevant regions from feature statistics.
- **S-SAA (Simplified Scale-Adaptive Attention)** — per-image learnable scale weighting across parallel dilated branches.
- **CDL (Contrastive Defect Learning)** — supervised contrastive objective with progressive warmup.

It is evaluated on a proprietary PEMFC electrode dataset (7 classes) and the public GC10-DET steel surface defect benchmark (11 classes).

---

## Results

| Dataset | Accuracy | Backbone baseline |
|---|---|---|
| PEMFC (752-image validation set) | **84.04%** | Swin V2-T 82.05% |
| GC10-DET (689-image validation set) | **87.07%** | Swin V2-T 82.76% |

Efficiency (single NVIDIA RTX A5000, 1024×1024): 30.3M params, 96.8 GMACs, 28.0 FPS.

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
pip install -r requirements.txt
```

---

## Key files

The SAAD-Net model and the scripts needed to reproduce the reported results:

| File | Purpose |
|---|---|
| `models/defect_lock_v2_improved.py` | **SAAD-Net model** (LAA, S-SAA, CDL modules) |
| `train_unified.py` | Main training / evaluation entry point (distributed, multi-GPU) |
| `gc10_crop_split.py` | GC10-DET preprocessing: horizontal bisection, Good-class generation, 80/20 stratified split |
| `measure_efficiency.py` | Params / FLOPs / FPS / peak-memory measurement |
| `eval/`, `utils/` | Evaluation and supporting utilities |

> **Note on imports.** `train_unified.py` imports the model as `from defect_lock_v2_improved import ...`, so make sure `defect_lock_v2_improved.py` is importable from the working directory (either run from the repository root with `models/` on the `PYTHONPATH`, or place `defect_lock_v2_improved.py` in the root). The repository also contains earlier development variants (`defect_lock.py`, `defect_lock_v2.py`, `saad_net_origin.py`, `gc10det_training.py`, alternative `train_*.py` scripts) that are **not** used for the reported results; the files listed above are the ones that reproduce the paper.

---

## Datasets

**PEMFC electrode dataset.** Proprietary, collected from a live fuel-cell manufacturing line; cannot be redistributed due to industrial confidentiality. Available from the corresponding author upon reasonable request.

**GC10-DET.** Publicly available at https://github.com/lvxiaoming2019/GC10-DET.

### GC10-DET preprocessing

`gc10_crop_split.py` reproduces the exact preprocessing described in the paper: each original 2048×1000 image is split along the vertical centerline into two 1024×1000 halves (each resized to 1024×1024). A half is labeled with a defect class if an annotated bounding box overlaps it by at least 30% of the box area; a half containing no annotated defect is labeled **Good**. Splitting is performed at the original-image level (both halves stay in the same split) with stratified sampling.

```bash
python gc10_crop_split.py \
    --data_dir /path/to/GC10-DET \
    --output_dir ./gc10_cropped \
    --train 0.8 --val 0.2 --seed 42
```

---

## Usage

### Training

```bash
# GC10-DET (11 classes), 8 GPUs, distributed
torchrun --nproc_per_node=8 train_unified.py \
    --data_dir ./gc10_cropped \
    --model defect_lock_v2_imp \
    --num_classes 11 \
    --img_size 1024 \
    --epochs 200 \
    --batch_size 4 \
    --lr 1e-4

# PEMFC (7 classes)
torchrun --nproc_per_node=8 train_unified.py \
    --data_dir /path/to/pemfc_processed \
    --model defect_lock_v2_imp \
    --num_classes 7 \
    --img_size 1024 \
    --epochs 200 \
    --batch_size 4 \
    --lr 5e-5
```

The full SAAD-Net model corresponds to `--model defect_lock_v2_imp` with the default config. Baselines in the paper are selected via `--model` (e.g. `swin_v2_t`, `vit_base`, `efficientnet_b4`, `resnet50`).

### Efficiency measurement

```bash
CUDA_VISIBLE_DEVICES=0 python measure_efficiency.py --img_size 1024
```

---

## Reproducing the paper

- Backbone: Swin Transformer V2-Tiny pretrained on ImageNet-22K.
- Evaluation aggregates metrics across all GPUs (`dist.all_reduce`) and reports on the **full validation set**.
- Random seed fixed (`seed=42`) for dataset splitting.
- Reported numbers correspond to the best-validation-accuracy checkpoint (PEMFC: Epoch 56; GC10-DET: Epoch 172).
- The PEMFC preprocessing pipeline (quadrant division with orientation normalization, Bilateral-CLAHE enhancement) is described in the paper; scripts are available from the corresponding author upon reasonable request.

---

## Citation

```bibtex
@article{joo2026saadnet,
  title   = {SAAD-Net: Scale-Adaptive Anomaly-aware Defect Network with Contrastive Learning for Multi-Class Defect Classification},
  author  = {Joo, In and Kim, Sung-Hoon and Seo, Mintaek and Ryu, Ga-Ae and Yoo, Kwan-Hee},
  journal = {Scientific Reports},
  year    = {2026}
}
```

---

## License

See `LICENSE`.
