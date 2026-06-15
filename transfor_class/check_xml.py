#!/usr/bin/env python3
"""Check actual XML structure and class names in GC10 dataset."""
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

DATA_DIR = '/home/cgac/jooin/data/gc10_cropped/voc'

for split in ['train', 'val']:
    ann_dir = Path(DATA_DIR) / split / 'annotations'
    xmls = sorted(ann_dir.rglob('*.xml'))
    print(f"\n{'='*60}")
    print(f"[{split}] - checking first 3 XMLs raw structure")
    print('='*60)

    all_classes = Counter()
    has_objects = 0

    for xml in xmls[:200]:
        root = ET.parse(xml).getroot()
        objs = root.findall('object')
        if objs:
            has_objects += 1
        for obj in objs:
            # Print ALL child tags to find the class tag name
            for child in obj:
                if child.tag not in ('bndbox', 'difficult', 'truncated', 'pose'):
                    all_classes[f"<{child.tag}>{child.text}</{child.tag}>"] += 1

    # Print first 3 XMLs fully
    for xml in xmls[:3]:
        print(f"\n--- {xml.name} ---")
        root = ET.parse(xml).getroot()
        objs = root.findall('object')
        print(f"  object count: {len(objs)}")
        for obj in objs[:2]:
            print(f"  object children: {[(c.tag, c.text) for c in obj if c.tag != 'bndbox']}")
            bb = obj.find('bndbox')
            if bb is not None:
                print(f"  bndbox: {[(c.tag, c.text) for c in bb]}")

    print(f"\n  XMLs with objects (first 200): {has_objects}/200")
    print(f"  All class-like tags found:")
    for tag, cnt in all_classes.most_common(20):
        print(f"    {cnt:4d}x  {tag}")