#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_v2_patch.py
==================
train_unified.py  SAAD-Net v2  .

Usage:
    python apply_v2_patch.py          #  ./train_unified.py 
    python apply_v2_patch.py --path /path/to/train_unified.py
"""

import re
import shutil
import argparse
from pathlib import Path


def patch_train_unified(target: str = './train_unified.py'):
    src = Path(target)
    if not src.exists():
        print(f"[ERROR] {src}  .")
        return

    # Backup
    bak = src.with_suffix('.py.bak_v2')
    shutil.copy(src, bak)
    print(f"[OK] Backup: {bak}")

    text = src.read_text(encoding='utf-8')

    # ------------------------------------------------------------------
    # PATCH 1: Import saad_net_v2 (insert after saad_net import block)
    # ------------------------------------------------------------------
    v2_import = '''
try:
    from saad_net_v2 import (
        SAADNetV2,
        get_saad_net_v2,
        SAADNetV2Loss,
        get_saad_net_v2_loss,
    )
    SAAD_NET_V2_AVAILABLE = True
except ImportError as e:
    SAAD_NET_V2_AVAILABLE = False
    print(f"Warning: saad_net_v2.py import failed - {e}")
'''

    # Insert after the saad_net import block
    if 'SAAD_NET_V2_AVAILABLE' not in text:
        anchor = "    SAAD_NET_AVAILABLE = True\nexcept ImportError as e:\n    SAAD_NET_AVAILABLE = False"
        if anchor in text:
            text = text.replace(anchor, anchor + v2_import, 1)
            print("[OK] PATCH 1: saad_net_v2 import added")
        else:
            # Fallback: insert before large_image_pipeline import
            anchor2 = "try:\n    from large_image_pipeline"
            text = text.replace(anchor2, v2_import + "\n" + anchor2, 1)
            print("[OK] PATCH 1: saad_net_v2 import added (fallback position)")
    else:
        print("[SKIP] PATCH 1: already applied")

    # ------------------------------------------------------------------
    # PATCH 2: build_detection_model() - add saad_net_v2_detection branch
    # ------------------------------------------------------------------
    v2_branch = '''
    # ---- SAAD-Net v2 ----
    if model_name == 'saad_net_v2_detection':
        if not SAAD_NET_V2_AVAILABLE:
            raise ImportError("saad_net_v2.py not found.")
        config = model_kwargs.pop('config', 'v2_default')
        return get_saad_net_v2(num_classes=num_classes, config=config, **model_kwargs)
'''

    if 'saad_net_v2_detection' not in text:
        anchor = "    raise ValueError(f\"Unknown detection model: {model_name}\")"
        text = text.replace(anchor, v2_branch + "\n" + anchor, 1)
        print("[OK] PATCH 2: build_detection_model() extended")
    else:
        print("[SKIP] PATCH 2: already applied")

    # ------------------------------------------------------------------
    # PATCH 3: argparse - add --saad_v2_config and model choice
    # ------------------------------------------------------------------
    v2_arg = '''
    # SAAD-Net v2 ablation config
    parser.add_argument('--saad_v2_config', type=str, default='v2_default',
                        choices=['v2_baseline', 'v2_ms_laa', 'v2_ms_laa_fda',
                                 'v2_default', 'v2_no_det', 'v2_light'],
                        help='SAAD-Net v2 ablation config (MS-LAA -> FDA -> Proto-PCA)')
'''

    if '--saad_v2_config' not in text:
        # Insert after --saad_pca_weight
        anchor = "    parser.add_argument('--saad_pca_weight'"
        idx = text.find(anchor)
        if idx != -1:
            # Find end of that argument line
            end = text.find('\n', idx)
            text = text[:end+1] + v2_arg + text[end+1:]
            print("[OK] PATCH 3a: --saad_v2_config arg added")
        else:
            print("[WARN] PATCH 3a: anchor not found, skipping --saad_v2_config")
    else:
        print("[SKIP] PATCH 3a: already applied")

    # Add saad_net_v2_detection to --model choices
    if 'saad_net_v2_detection' not in text:
        anchor = "'saad_net_detection',"
        text = text.replace(anchor, "'saad_net_detection',\n                            'saad_net_v2_detection',", 1)
        print("[OK] PATCH 3b: saad_net_v2_detection added to --model choices")
    else:
        print("[SKIP] PATCH 3b: already applied")

    # ------------------------------------------------------------------
    # PATCH 4: train_detection() - handle saad_net_v2_detection model + loss
    # ------------------------------------------------------------------
    v2_model_block = '''
    # -- Model + Loss (v1 or v2) ------------------------------------------
    is_v2 = (args.model == 'saad_net_v2_detection')
    cfg   = args.saad_v2_config if is_v2 else args.saad_config

    model = build_detection_model(
        args.model,
        num_classes  = args.num_classes,
        model_kwargs = {
            'model_size': args.model_size,
            'config'    : cfg,
            'pretrained': True,
        },
    )

    if is_v2:
        criterion = get_saad_net_v2_loss(
            num_classes     = args.num_classes,
            lambda_det      = args.saad_det_weight,
            lambda_pca      = args.saad_pca_weight,
            focal_gamma     = 2.0,
            label_smoothing = 0.1,
        )
        print_rank0(f"Loss: SAADNetV2Loss  det={args.saad_det_weight}  pca={args.saad_pca_weight}")
    else:
        criterion = get_saad_net_loss(
            num_classes      = args.num_classes,
            lambda_det       = args.saad_det_weight,
            lambda_pca       = args.saad_pca_weight,
            focal_gamma      = 2.0,
            label_smoothing  = 0.1,
        )
        print_rank0(f"Loss: SAADNetLoss  det={args.saad_det_weight}  pca={args.saad_pca_weight}")
'''

    if 'is_v2 = (args.model ==' not in text:
        # Replace the old fixed model + criterion block in train_detection()
        old_model_block = """    # -- Model -----------------------------------------------------------
    model = build_detection_model(
        'saad_net_detection',
        num_classes  = args.num_classes,
        model_kwargs = {
            'model_size': args.model_size,
            'config'    : args.saad_config,
            'pretrained': True,
        },
    )

    if distributed and args.model not in ['saad_net_detection']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # -- Loss ------------------------------------------------------------
    criterion = get_saad_net_loss(
        num_classes      = args.num_classes,
        lambda_det       = args.saad_det_weight,
        lambda_pca       = args.saad_pca_weight,
        focal_gamma      = 2.0,
        label_smoothing  = 0.1,
    )
    print_rank0(f"SAADNetLoss: focal=1.0 det={args.saad_det_weight} pca={args.saad_pca_weight}")"""

        new_model_block = (v2_model_block + """
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
""")

        if old_model_block in text:
            text = text.replace(old_model_block, new_model_block, 1)
            print("[OK] PATCH 4: train_detection() model+loss block updated")
        else:
            print("[WARN] PATCH 4: could not find exact model block. "
                  "Please apply manually (see train_unified_v2_patch.py)")
    else:
        print("[SKIP] PATCH 4: already applied")

    # ------------------------------------------------------------------
    # PATCH 5: auto-set task for saad_net_v2_detection
    # ------------------------------------------------------------------
    if 'saad_net_v2_detection' not in text.split('auto-setting')[0] if 'auto-setting' in text else True:
        old_auto = """    if args.model == 'saad_net_detection' and args.task == 'classification':
        args.task = 'detection'
        print_rank0("[INFO] model=saad_net_detection -> auto-setting task to detection")"""

        new_auto = """    if args.model in ['saad_net_detection', 'saad_net_v2_detection'] and args.task == 'classification':
        args.task = 'detection'
        print_rank0(f"[INFO] model={args.model} -> auto-setting task to detection")"""

        if old_auto in text:
            text = text.replace(old_auto, new_auto, 1)
            print("[OK] PATCH 5: auto-task for saad_net_v2_detection added")
        else:
            print("[WARN] PATCH 5: auto-task anchor not found")
    else:
        print("[SKIP] PATCH 5: already applied")

    # ------------------------------------------------------------------
    # PATCH 6: print header in train_detection - show v1/v2
    # ------------------------------------------------------------------
    old_header = """    print_rank0(f"SAAD-Net Detection Training")
    print_rank0(f"  config      : {args.saad_config}")"""

    new_header = """    is_v2 = getattr(args, 'model', '') == 'saad_net_v2_detection'
    _cfg  = getattr(args, 'saad_v2_config', args.saad_config) if is_v2 else args.saad_config
    print_rank0(f"SAAD-Net {'v2' if is_v2 else 'v1'} Detection Training")
    print_rank0(f"  config      : {_cfg}")"""

    if old_header in text and 'is_v2 = getattr' not in text:
        text = text.replace(old_header, new_header, 1)
        print("[OK] PATCH 6: print header updated")
    else:
        print("[SKIP] PATCH 6: already applied or not found")

    # Write
    src.write_text(text, encoding='utf-8')
    print(f"\n[DONE] Patched file saved: {src}")
    print("Run quick test:")
    print("  python train_unified.py --task detection --model saad_net_v2_detection "
          "--saad_v2_config v2_default --data_dir ./gc10_cropped/voc --num_classes 11 "
          "--img_size 512 --batch_size 2 --epochs 1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./train_unified.py',
                        help='Path to train_unified.py')
    args = parser.parse_args()
    patch_train_unified(args.path)