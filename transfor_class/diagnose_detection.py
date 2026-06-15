#!/usr/bin/env python3
"""
Quick diagnostic for mAP=0 / Images=0 issue.
Run: python diagnose_detection.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pathlib import Path
import xml.etree.ElementTree as ET

DATA_DIR  = '/home/cgac/jooin/data/gc10_cropped/voc'
NUM_CLASS = 11
IMG_SIZE  = 1024

print("=" * 60)
print("STEP 1: Check data directory structure")
print("=" * 60)
for split in ['train', 'val']:
    img_dir = Path(DATA_DIR) / split / 'images'
    ann_dir = Path(DATA_DIR) / split / 'annotations'
    imgs    = sorted(img_dir.rglob('*')) if img_dir.exists() else []
    imgs    = [p for p in imgs if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')]
    xmls    = sorted(ann_dir.rglob('*.xml')) if ann_dir.exists() else []
    print(f"\n[{split}]")
    print(f"  images dir exists : {img_dir.exists()}  ({img_dir})")
    print(f"  annotations dir   : {ann_dir.exists()}  ({ann_dir})")
    print(f"  image count       : {len(imgs)}")
    print(f"  xml count         : {len(xmls)}")
    if xmls:
        root = ET.parse(xmls[0]).getroot()
        objs = root.findall('object')
        print(f"  first xml ({xmls[0].name}): {len(objs)} objects")
        for o in objs[:3]:
            n = o.find('name') or o.find('n')
            print(f"    class: {n.text.strip() if n is not None else '?'}")

print()
print("=" * 60)
print("STEP 2: Check model output keys (1 batch on CPU)")
print("=" * 60)

device = 'cpu'
try:
    from transfor_class.saad_net_v3 import get_saad_net_v2
    model = get_saad_net_v2(num_classes=NUM_CLASS, config='v2_baseline', pretrained=False)
    model.eval()

    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(x)

    print(f"Output type   : {type(out)}")
    if isinstance(out, dict):
        print(f"Output keys   : {list(out.keys())}")
        if 'det_outputs' in out:
            det = out['det_outputs']
            print(f"det_outputs   : type={type(det)}, len={len(det) if det else 0}")
            if det:
                print(f"  level 0 keys: {list(det[0].keys())}")
                for k, v in det[0].items():
                    print(f"  {k}: {v.shape if hasattr(v,'shape') else v}")
            else:
                print("  *** det_outputs is EMPTY LIST -> mAP will always be 0 ***")
        else:
            print("  *** 'det_outputs' KEY MISSING from model output ***")
            print("  -> v2_baseline config may not have detection head enabled")
        if 'logits' in out:
            print(f"logits shape  : {out['logits'].shape}")
    else:
        print(f"  *** output is not dict: {out}")

except Exception as e:
    print(f"Model error: {e}")
    import traceback; traceback.print_exc()

print()
print("=" * 60)
print("STEP 3: Check saad_net_v2 v2_baseline config")
print("=" * 60)
try:
    from transfor_class.saad_net_v3 import get_saad_net_v2
    import inspect
    src = inspect.getsource(get_saad_net_v2)
    # Find v2_baseline config block
    idx = src.find("'v2_baseline'")
    if idx >= 0:
        print(src[idx:idx+400])
    else:
        print("v2_baseline config not found in source")
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("STEP 4: Summary & Fix")
print("=" * 60)
print("""
If STEP 2 shows 'det_outputs is EMPTY LIST':
  -> v2_baseline config has use_det_head=False or det_head not building properly
  -> Fix: set use_det_head=True in v2_baseline config in saad_net_v2.py

If STEP 1 shows xml count=0:
  -> No annotation files in val/annotations/
  -> Possible: different folder structure (annotations might be in train only)
  -> Check: find /home/cgac/jooin/data/gc10_cropped -name "*.xml" | head -5
""")