#!/usr/bin/env python
# FIX BUG6: single import os at top, before everything else
import os
os.environ['NCCL_BLOCKING_WAIT']      = '1'
os.environ['NCCL_TIMEOUT']            = '1800'
os.environ['NCCL_DEBUG']              = 'INFO'
os.environ['NCCL_P2P_DISABLE']        = '1'
os.environ['NCCL_IB_DISABLE']         = '1'
os.environ['CUDA_LAUNCH_BLOCKING']    = '0'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

"""
train_saad_ultra.py
====================
Ultralytics RT-DETR-L + SAAD-Net novel contributions

Novel #1 - AnomalyAwareAIFI  : AIFI layer replacement
Novel #2 - FDA                : backbone feature hook
Novel #3 - ProtoPCA           : contrastive loss via patched model.forward

Single GPU:
    python train_saad_ultra.py --data data.yaml --config v3_default

Multi GPU (torchrun):
    torchrun --nproc_per_node=4 train_saad_ultra.py --data data.yaml --config v3_default

Ablation (all 4 configs sequentially):
    python train_saad_ultra.py --data data.yaml --run_ablation
"""

# FIX BUG6: 'import os' removed here (already at line 2)
import argparse
import json
import torch
from pathlib import Path

from saad_net_v3 import (
    build_saad_model,
    make_saad_trainer,
    FDAHookManager,
    PrototypePCA,
)

CONFIGS = ['v3_baseline', 'v3_ms_laa', 'v3_ms_laa_fda', 'v3_default']


# ============================================================================
# Callbacks
# ============================================================================

def make_callbacks(saad_components: dict, args):
    """
    Returns dict of Ultralytics-compatible callbacks.
    Each callback receives the trainer (or validator) as argument.
    """
    fda_manager = saad_components.get('fda_manager')
    proto_pca   = saad_components.get('proto_pca')
    qf_capture  = saad_components.get('qf_capture')
    cfg         = saad_components.get('cfg', {})

    callbacks = {}

    # ------------------------------------------------------------------
    # on_train_start: move FDA/PCA to correct device, log config
    # ------------------------------------------------------------------
    def on_train_start(trainer):
        device = next(trainer.model.parameters()).device
        if fda_manager:
            fda_manager.to(device).train()
        if proto_pca:
            proto_pca.to(device).train()
        print(f"\n[SAAD] Training started  config={saad_components['config']}")
        print(f"       device={device}  "
              f"aifi={cfg.get('use_aifi')}  "
              f"fda={cfg.get('use_fda')}  "
              f"pca={cfg.get('use_pca')}")

    callbacks['on_train_start'] = on_train_start

    # ------------------------------------------------------------------
    # on_train_epoch_start: log progressive PCA warmup weight
    # FIX BUG9: guard max(args.pca_warmup, 1) to prevent ZeroDivisionError
    # ------------------------------------------------------------------
    def on_train_epoch_start(trainer):
        epoch = trainer.epoch
        if proto_pca is not None and args.pca_warmup > 0:
            # FIX BUG9: max(args.pca_warmup, 1) - already guaranteed by args.pca_warmup > 0
            # but also guard here for safety
            w_t = args.w_pca * min(epoch / max(args.pca_warmup, 1), 1.0)
            if epoch <= args.pca_warmup:
                print(f"[SAAD] Epoch {epoch + 1}  PCA warmup  w={w_t:.4f}/{args.w_pca}")

    callbacks['on_train_epoch_start'] = on_train_epoch_start

    # ------------------------------------------------------------------
    # on_val_start / on_val_end: switch FDA/PCA eval/train mode
    # ------------------------------------------------------------------
    def on_val_start(validator):
        if fda_manager:
            fda_manager.eval()
        if proto_pca:
            proto_pca.eval()

    def on_val_end(validator):
        if fda_manager:
            fda_manager.train()
        if proto_pca:
            proto_pca.train()

    callbacks['on_val_start'] = on_val_start
    callbacks['on_val_end']   = on_val_end

    # ------------------------------------------------------------------
    # on_fit_epoch_end: fires AFTER validation
    # FIX BUG7: moved mAP50 logging here (was in on_train_epoch_end which
    #           fires BEFORE validation, giving stale previous-epoch metrics)
    # FIX BUG8: only save saad_extra.pt when FDA or PCA state actually exists
    # ------------------------------------------------------------------
    def on_fit_epoch_end(trainer):
        # FIX BUG7: read metrics here, after validation has completed
        metrics = trainer.metrics or {}
        map50   = metrics.get('metrics/mAP50(B)', 0.0)
        map5095 = metrics.get('metrics/mAP50-95(B)', 0.0)
        if map50 > 0:
            print(f"[SAAD] Epoch {trainer.epoch + 1}  "
                  f"mAP50={map50:.4f}  mAP50-95={map5095:.4f}")

        # FIX BUG8: build extra dict first; only save if non-empty
        extra = {}
        if fda_manager and fda_manager.fda_modules:
            extra['fda_state'] = fda_manager.state_dict()
        if proto_pca is not None:
            extra['pca_state'] = proto_pca.state_dict()

        # Only write file when there is actual SAAD state to persist
        if extra:
            extra['config'] = saad_components['config']
            extra['epoch']  = trainer.epoch
            save_path = Path(trainer.save_dir) / 'saad_extra.pt'
            torch.save(extra, save_path)

    callbacks['on_fit_epoch_end'] = on_fit_epoch_end

    # ------------------------------------------------------------------
    # on_train_end: final summary + JSON report
    # ------------------------------------------------------------------
    def on_train_end(trainer):
        metrics  = trainer.metrics or {}
        map50    = metrics.get('metrics/mAP50(B)', 0.0)
        save_dir = Path(trainer.save_dir)
        print(f"\n[SAAD] Training complete  best mAP50={map50:.4f}")
        print(f"       weights -> {save_dir / 'weights/best.pt'}")
        if (save_dir / 'saad_extra.pt').exists():
            print(f"       extras  -> {save_dir / 'saad_extra.pt'}")

        summary = {
            'config':  saad_components['config'],
            'mAP50':   map50,
            'mAP5095': metrics.get('metrics/mAP50-95(B)', 0.0),
        }
        (save_dir / 'saad_summary.json').write_text(json.dumps(summary, indent=2))

    callbacks['on_train_end'] = on_train_end

    return callbacks


# ============================================================================
# Main training function
# ============================================================================

def train(args):
    # ---- Build model + inject novel components ----
    saad = build_saad_model(
        weights=args.weights,
        config=args.config,
        num_classes=args.num_classes,
        device=args.device or None,
    )
    model       = saad['model']
    fda_manager = saad['fda_manager']
    proto_pca   = saad['proto_pca']

    # ---- Build custom trainer class ----
    SAADTrainer = make_saad_trainer(saad, w_pca=args.w_pca, pca_warmup=args.pca_warmup)

    # ---- Register callbacks ----
    cbs = make_callbacks(saad, args)
    for event, fn in cbs.items():
        model.add_callback(event, fn)

    # ---- Training hyperparameters ----
    train_cfg = dict(
        data          = args.data,
        epochs        = args.epochs,
        imgsz         = args.img_size,
        batch         = args.batch_size,
        device        = args.device,
        lr0           = args.lr,
        lrf           = 0.01,
        warmup_epochs = args.warmup,
        patience      = args.patience,
        project       = args.project,
        name          = args.config,
        optimizer     = 'AdamW',
        weight_decay  = args.weight_decay,
        cos_lr        = True,
        amp           = True,
        verbose       = True,
        plots         = True,
    )
    # save_dir only if explicitly set (avoid overriding Ultralytics default)
    if args.save_dir:
        train_cfg['save_dir'] = args.save_dir

    print(f"\n{'='*60}")
    print(f"SAAD-Net  config={args.config}  weights={args.weights}")
    print(f"data={args.data}  epochs={args.epochs}  batch={args.batch_size}")
    print(f"patience={args.patience}  w_pca={args.w_pca}  pca_warmup={args.pca_warmup}")
    print(f"{'='*60}\n")

    results = model.train(**train_cfg, trainer=SAADTrainer)

    # ---- Cleanup hooks ----
    if fda_manager:
        fda_manager.remove()
    if saad.get('qf_capture'):
        saad['qf_capture'].remove()

    return results


# ============================================================================
# Ablation runner: v3_baseline -> v3_ms_laa -> v3_ms_laa_fda -> v3_default
# ============================================================================

def run_ablation(args):
    results  = {}
    base_dir = args.save_dir or './runs/saad_ablation'

    for cfg_name in CONFIGS:
        print(f"\n{'#'*60}")
        print(f"# Ablation: {cfg_name}")
        print(f"{'#'*60}")
        args.config   = cfg_name
        args.save_dir = str(Path(base_dir) / cfg_name)
        res = train(args)

        map50 = 0.0
        try:
            map50 = float(res.results_dict.get('metrics/mAP50(B)', 0.0))
        except Exception:
            pass
        results[cfg_name] = map50

    # Print table
    print(f"\n{'='*50}")
    print(f"Ablation Results")
    print(f"{'='*50}")
    print(f"{'Config':<22} {'mAP50':>8}")
    print(f"{'-'*32}")
    for cfg_name, v in results.items():
        print(f"{cfg_name:<22} {v:>8.4f}")

    out = Path(base_dir) / 'ablation_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='SAAD-Net Ultralytics Trainer')

    # Data
    p.add_argument('--data',         type=str, required=True,
                   help='Path to data.yaml (Ultralytics format)')

    # Model
    p.add_argument('--weights',      type=str, default='rtdetr-l.pt',
                   help='Base weights: rtdetr-l.pt or path to fine-tuned .pt')
    p.add_argument('--config',       type=str, default='v3_default',
                   choices=CONFIGS,
                   help='SAAD-Net ablation config')
    p.add_argument('--num_classes',  type=int, default=-1,
                   help='-1 = auto-detect from data.yaml')

    # Training
    p.add_argument('--img_size',     type=int,   default=640)
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup',       type=int,   default=3,
                   help='Warmup epochs')
    p.add_argument('--patience',     type=int,   default=50,
                   help='Early stopping patience (0 = disable)')
    p.add_argument('--device',       type=str,   default='',
                   help='cuda device: 0  |  0,1,2,3  |  cpu  (blank = auto)')

    # Novel #3 ProtoPCA
    p.add_argument('--w_pca',        type=float, default=0.1,
                   help='ProtoPCA loss weight')
    p.add_argument('--pca_warmup',   type=int,   default=10,
                   help='Epochs to linearly ramp PCA weight 0 -> w_pca  (0 = no warmup)')

    # Output
    p.add_argument('--save_dir',     type=str, default='',
                   help='Override Ultralytics default save directory')
    p.add_argument('--project',      type=str, default='saad_net')

    # Ablation
    p.add_argument('--run_ablation', action='store_true',
                   help='Run all 4 ablation configs sequentially')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.run_ablation:
        run_ablation(args)
    else:
        train(args)