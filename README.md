# SAACD-Net: Scale-Adaptive Anomaly-aware Contrastive Defect Learning Network

Code for the manuscript **"Multi-Class Defect Classification Using a Swin Transformer with Scale-Adaptive Attention and Contrastive Learning"** (under review).

> The paper calls the model **SAACD-Net**. The repository and the code keep the earlier name SAAD-Net: the model class is `SAADNet` (an alias of `DefectLoCKv2Improved`) and the training script selects it with `--model defect_lock_v2_imp`.

SAACD-Net is a multi-class defect classifier for high-resolution inspection images. It keeps a Swin Transformer V2-Tiny backbone and adds the established CBAM module and three components:

- **LAA (Lightweight Anomaly-Aware Attention)**: computes an anomaly map from feature statistics and amplifies its high-scoring regions with a learnable residual gain.
- **S-SAA (Simplified Scale-Adaptive Attention)**: fuses parallel dilated branches (d = 1, 2, 4) and a global-context branch with per-image weights.
- **CDL (Contrastive Defect Learning)**: a supervised contrastive objective introduced with a progressive warmup.

---

## Results

Mean ± standard deviation over three independent runs per model, with identical hyperparameters and no fixed training seed. The checkpoint of each run is selected on the validation split; the held-out test splits are used for neither training nor checkpoint selection.

| Dataset | Split | SAACD-Net | Swin V2-T backbone | Runs |
|---|---|---|---|---|
| Contact lens (5 defect classes) | held-out test, 3,217 images | **96.22 ± 0.72%** | 94.87 ± 0.91% | 3 |
| GC10-DET (11 classes) | held-out test, 701 images | **84.35 ± 1.51%** | 80.70 ± 3.07% | 3 |

The other compared models, the ablation study and the significance tests are reported in the paper.

Efficiency (one NVIDIA RTX A5000, 1024×1024 input, batch size 1, eleven-class head): 30.3M parameters, 96.8 GMACs, 28.0 FPS, 427 MB peak memory.

---

## Requirements

Tested with Python 3.9, PyTorch 2.6 (CUDA 12.4) and torchvision 0.21.

```bash
pip install torch==2.6.0 torchvision==0.21.0 timm==1.0.15 numpy scipy scikit-learn \
    opencv-python pillow matplotlib seaborn tqdm fvcore
```

`timm` is needed only for the ViT-Base, DeiT3, PVT-v2 and CAFormer baselines. `transfor_class/requirements.txt` is a full export of the training environment, kept for reference.

---

## Repository layout

| Path | Purpose |
|---|---|
| `transfor_class/defect_lock_v2_improved.py` | **SAACD-Net model** (CBAM, LAA, S-SAA, CDL) and its training loss |
| `transfor_class/train_unified.py` | Training entry point (distributed data parallel, multi-GPU) |
| `transfor_class/eval/comprehensive_evaluation.py` | Evaluation of a saved SAACD-Net checkpoint on a class-folder split |
| `results/gc10/` | Training history and confusion matrix of the representative GC10-DET run |

`train_unified.py` imports the model from the `defect_lock_v2_improved.py` file next to it, so run it from `transfor_class/`. The other model and training files in the repository (including `models/`, `defect_lock.py`, `defect_lock_v2.py`, `saad_net_origin.py`, `train_saad_v3.py`, the other `train_*.py` scripts and `run_gc10_benchmark.sh`) are earlier development versions and were **not** used for the reported results.

---

## Datasets

**Contact lens dataset.** Proprietary images from a contact lens production line; they cannot be redistributed due to industrial confidentiality. The dataset holds 21,443 single-label lens images of 1000×1000 pixels in five defect classes (`position`, `bubble`, `color`, `edge`, `etc`). The split is made first, at the level of lot–position groups (no group contributes images to more than one split), giving 15,010 training, 3,216 validation and 3,217 test images. Each image is then enhanced independently with Bilateral-CLAHE: a bilateral filter (d = 9, σ_color = 75, σ_space = 75) followed by CLAHE (clip limit 2.0, 8×8 tiles) on the lightness channel in CIELAB space.

**GC10-DET.** Public, available at https://github.com/lvxiaoming2019/GC10-DET. Each ~2048×1000 image is split along the vertical centerline into two halves. A half inherits every annotated box that places at least 30% of its area within it and takes the class of its largest annotated defect (objects flagged difficult are excluded); a half with no remaining defect is labeled `good`, giving 11 classes. The 70/15/15 train/validation/test split is stratified and made at the original-image level before bisection (seed 42), giving 3,224 training, 689 validation and 701 test images.

The lens split and enhancement scripts and the GC10-DET preprocessing script are not part of this repository; they are available from the corresponding author upon reasonable request.

Both datasets are read from class folders:

```
DATA_DIR/
  train/<class>/*.png
  val/<class>/*.png
  test/<class>/*.png
```

`train_unified.py` reads `train/` and `val/`. The `test/` split is used only for the final evaluation.

---

## Training

```bash
cd transfor_class

# Contact lens dataset (5 classes), 8 GPUs
torchrun --nproc_per_node=8 train_unified.py \
    --data_dir /path/to/lens_split_enhanced \
    --model defect_lock_v2_imp --defect_v2_config default \
    --num_classes 5 --img_size 1024 --batch_size 8 \
    --epochs 200 --patience 20 --warmup 5 --lr 5e-5 \
    --save_dir runs/lens_saacd_r1

# GC10-DET (11 classes), 4 GPUs
torchrun --nproc_per_node=4 train_unified.py \
    --data_dir /path/to/gc10_halves \
    --model defect_lock_v2_imp --defect_v2_config default \
    --num_classes 11 --img_size 1024 --batch_size 4 \
    --epochs 200 --patience 20 --warmup 5 --lr 1e-4 \
    --save_dir runs/gc10_saacd_r1
```

- `--batch_size` is per GPU (total 64 on the lens dataset, 16 on GC10-DET). `--lr` is a base rate that the script multiplies by the number of GPUs.
- The script saves the checkpoint with the best validation accuracy as `best_model.pth` and stops after 20 epochs without improvement. Training uses mixed precision (FP16).
- SAACD-Net loss: focal loss (γ = 2.0, label smoothing 0.1); supervised contrastive loss (weight 0.1, temperature 0.07, linear warmup over 10 epochs); S-SAA balance term (weight 0.1), the KL divergence that pulls the branch weights toward the uniform allocation. Every class of the two datasets receives this uniform target. The loss also contains a size table keyed by other class names, which is not used here; keep class folder names different from its keys.
- Training-time augmentation and all hyperparameters are listed in the paper.

### Ablation configurations (contact lens dataset)

| Row in the paper | `--defect_v2_config` |
|---|---|
| Baseline (Swin V2-T, modules off) | `baseline` |
| + CBAM | `cbam_only` |
| + LAA | `cbam_laa` |
| + S-SAA (still without CDL) / Full model without CDL | `attention_only` |
| Full model | `default` |
| Full model without S-SAA | `cbam_laa_cdl` |
| Full model without Bilateral-CLAHE | `default`, trained on the unenhanced split |

### Compared models

Selected with `--model` (no `--defect_v2_config`): `swin_v2_t`, `resnet50`, `efficientnet_b4` and `vit_base` on the lens dataset; `swin_v2_t`, `vit_base`, `deit3_small`, `pvt_v2_b2` and `caformer_s18` on GC10-DET. All use the same split, input size, optimizer and early-stopping rule as SAACD-Net, with focal loss (γ = 2.0) and no label smoothing.

---

## Evaluation on the test split

The test split is evaluated once per run with `best_model.pth`, using the validation preprocessing (resize to 1024×1024, ImageNet normalization):

```bash
cd transfor_class
PYTHONPATH=. python eval/comprehensive_evaluation.py \
    --checkpoint runs/lens_saacd_r1/best_model.pth \
    --data_dir /path/to/lens_split_enhanced/test \
    --img_size 1024 --batch_size 2 --output_dir eval_lens_r1
```

The evaluation script rebuilds the full SAACD-Net model.

---

## Notes for reproducing the paper

- Backbone: torchvision Swin Transformer V2-Tiny with ImageNet-1K weights.
- Every configuration was run three times; the paper reports mean ± standard deviation and two-sided Welch's *t*-tests on test accuracy.
- Per-class tables and confusion matrices in the paper come from one representative run, the run closest to the three-run test mean: 96.24% on the lens test split and 84.59% on the GC10-DET test split (checkpoint from epoch 172).

---

## Citation

```bibtex
@article{joo2026saacdnet,
  title   = {Multi-Class Defect Classification Using a Swin Transformer with Scale-Adaptive Attention and Contrastive Learning},
  author  = {Joo, In and Kim, Sung-Hoon and Ryu, Ga-Ae and Yoo, Kwan-Hee},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Corresponding author: Kwan-Hee Yoo (khyoo@chungbuk.ac.kr).

---

## License

See `transfor_class/LICENSE`.
