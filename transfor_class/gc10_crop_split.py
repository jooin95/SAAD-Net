#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GC10-DET Horizontal Crop Splitter
===================================
원본 2048×1000 이미지를 가로 2분할 → 1024×1000 × 2장

분할 로직:
  ┌──────────────────────────────────────────┐
  │  LEFT  (x: 0 ~ 1024)  │  RIGHT (x: 1024 ~ 2048) │
  └──────────────────────────────────────────┘

  각 절반마다 XML BBox를 확인하여:
    - BBox가 해당 절반에 면적의 30% 이상 걸쳐 있으면 → 그 절반에 포함 (좌표 클리핑)
    - 해당 절반에 BBox가 하나도 없으면              → "good" (정상) 클래스로 저장
    - BBox가 있으면                                 → 해당 defect 클래스로 저장

결과:
  - 데이터 2배 증가
  - Good 데이터 자동 생성 (defect가 없는 절반)
  - Classification / YOLO / VOC 포맷 동시 출력

출력 클래스 (11개):
  0: punching_hole   1: welding_line   2: crescent_gap
  3: water_spot      4: oil_spot       5: silk_spot
  6: inclusion       7: rolled_pit     8: crease
  9: waist_folding   10: good

사용법:
    python gc10_crop_split.py --data_dir /path/to/GC10-DET --output_dir ./gc10_cropped

    # 비율 조정
    python gc10_crop_split.py --data_dir /path/to/GC10-DET --output_dir ./gc10_cropped \\
        --train 0.8 --val 0.2

    # 특정 포맷만
    python gc10_crop_split.py --data_dir /path/to/GC10-DET --output_dir ./gc10_cropped \\
        --format yolo

    # 겹침 임계값 조정 (bbox 면적의 몇 % 이상이어야 포함)
    python gc10_crop_split.py --data_dir /path/to/GC10-DET --output_dir ./gc10_cropped \\
        --overlap_thresh 0.1
"""

import os
import json
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow 없음 — 이미지 크롭은 OpenCV로 대체됩니다.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ============================================================================
# 클래스 정의 (good 추가)
# ============================================================================

GC10_CLASSES = {
    1:  'punching_hole',
    2:  'welding_line',
    3:  'crescent_gap',
    4:  'water_spot',
    5:  'oil_spot',
    6:  'silk_spot',
    7:  'inclusion',
    8:  'rolled_pit',
    9:  'crease',
    10: 'waist_folding',
}

GC10_CLASS_KO = {
    1:  '펀칭홀',
    2:  '용접선',
    3:  '월아완',
    4:  '수적',
    5:  '유적',
    6:  '실크반점',
    7:  '개재물',
    8:  '압연구덩이',
    9:  '주름',
    10: '허리접힘',
}

# good 클래스 포함한 최종 클래스 목록 (0-indexed)
ALL_CLASSES  = list(GC10_CLASSES.values()) + ['good']
CLASS_TO_IDX = {name: i for i, name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(ALL_CLASSES)}
GOOD_IDX     = CLASS_TO_IDX['good']   # = 10

# 원본 folder 번호 → class name
FOLDER_TO_CLASS = {k: v for k, v in GC10_CLASSES.items()}


# ============================================================================
# 1. XML 파싱
# ============================================================================

def parse_voc_xml(xml_path: str) -> Optional[Dict]:
    """Pascal VOC XML → dict"""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return None

    size   = root.find('size')
    width  = int(size.find('width').text)
    height = int(size.find('height').text)
    folder = root.find('folder').text.strip()

    objects = []
    for obj in root.findall('object'):
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bb = obj.find('bndbox')
        xmin = max(0,     int(float(bb.find('xmin').text)))
        ymin = max(0,     int(float(bb.find('ymin').text)))
        xmax = min(width, int(float(bb.find('xmax').text)))
        ymax = min(height,int(float(bb.find('ymax').text)))

        if xmax <= xmin or ymax <= ymin:
            continue

        try:
            folder_int = int(folder)
            class_name = FOLDER_TO_CLASS.get(folder_int, 'punching_hole')
        except ValueError:
            class_name = 'punching_hole'

        objects.append({
            'class_name': class_name,
            'class_idx' : CLASS_TO_IDX[class_name],
            'xmin': xmin, 'ymin': ymin,
            'xmax': xmax, 'ymax': ymax,
            'difficult': difficult,
        })

    return {'width': width, 'height': height, 'folder': folder, 'objects': objects}


# ============================================================================
# 2. 가로 2분할 BBox 처리
# ============================================================================

def split_objects_by_crop(
    objects: List[Dict],
    img_w: int,
    img_h: int,
    overlap_thresh: float = 0.3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    원본 이미지의 BBox 목록을 Left / Right 절반으로 분리

    Args:
        objects        : parse_voc_xml 결과의 objects 리스트
        img_w          : 원본 이미지 너비 (2048)
        img_h          : 원본 이미지 높이 (1000)
        overlap_thresh : 해당 절반에 bbox 면적의 몇 % 이상 겹쳐야 포함 (default 0.3)

    Returns:
        left_objs  : Left 절반(0~img_w//2)에 해당하는 BBox 목록 (좌표는 절반 기준으로 클리핑)
        right_objs : Right 절반(img_w//2~img_w)에 해당하는 BBox 목록 (x좌표 shift)

    각 obj dict:
        {class_name, class_idx, xmin, ymin, xmax, ymax, difficult,
         orig_xmin, orig_xmax}  ← 원본 좌표 보존
    """
    half_w = img_w // 2  # 1024
    left_objs  = []
    right_objs = []

    for obj in objects:
        xmin, ymin = obj['xmin'], obj['ymin']
        xmax, ymax = obj['xmax'], obj['ymax']
        orig_area  = max(1, (xmax - xmin) * (ymax - ymin))

        # ── LEFT 절반 (x: 0 ~ half_w) ──────────────────────────────────
        l_xmin = max(0,      xmin)
        l_xmax = min(half_w, xmax)
        if l_xmax > l_xmin:
            inter_area = (l_xmax - l_xmin) * (ymax - ymin)
            overlap    = inter_area / orig_area
            if overlap >= overlap_thresh:
                left_objs.append({
                    **obj,
                    'xmin': l_xmin,
                    'xmax': l_xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'orig_xmin': xmin,
                    'orig_xmax': xmax,
                })

        # ── RIGHT 절반 (x: half_w ~ img_w), x좌표를 half_w 만큼 shift ──
        r_xmin = max(half_w, xmin)
        r_xmax = min(img_w,  xmax)
        if r_xmax > r_xmin:
            inter_area = (r_xmax - r_xmin) * (ymax - ymin)
            overlap    = inter_area / orig_area
            if overlap >= overlap_thresh:
                right_objs.append({
                    **obj,
                    'xmin': r_xmin - half_w,   # shift
                    'xmax': r_xmax - half_w,
                    'ymin': ymin,
                    'ymax': ymax,
                    'orig_xmin': xmin,
                    'orig_xmax': xmax,
                })

    return left_objs, right_objs


def determine_crop_class(objs: List[Dict]) -> Tuple[str, int]:
    """
    절반 crop의 클래스 결정:
    - BBox 없음           → good (10)
    - BBox 1종            → 해당 defect class
    - BBox 여러 종 혼재   → 면적 최대 defect class (multi-label이 아닌 single-label)
    """
    if not objs:
        return 'good', GOOD_IDX

    # difficult=1 제외
    valid = [o for o in objs if not o['difficult']]
    if not valid:
        return 'good', GOOD_IDX

    # 클래스별 총 면적 집계
    area_per_class = defaultdict(float)
    for o in valid:
        area = (o['xmax'] - o['xmin']) * (o['ymax'] - o['ymin'])
        area_per_class[o['class_name']] += area

    best_class = max(area_per_class, key=lambda k: area_per_class[k])
    return best_class, CLASS_TO_IDX[best_class]


# ============================================================================
# 3. 이미지 크롭 저장
# ============================================================================

def crop_and_save(img_path: str, crop_xmin: int, crop_xmax: int,
                  dst_path: Path) -> bool:
    """이미지를 crop_xmin~crop_xmax 구간으로 크롭하여 저장"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PIL:
        try:
            img = Image.open(img_path)
            # grayscale → RGB 변환
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            w, h = img.size
            cropped = img.crop((crop_xmin, 0, crop_xmax, h))
            cropped.save(str(dst_path), quality=95)
            return True
        except Exception as e:
            print(f"  [PIL ERROR] {img_path}: {e}")

    if HAS_CV2:
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False
            cropped = img[:, crop_xmin:crop_xmax]
            cv2.imwrite(str(dst_path), cropped)
            return True
        except Exception as e:
            print(f"  [CV2 ERROR] {img_path}: {e}")

    print(f"  [ERROR] PIL/OpenCV 모두 없음. pip install pillow")
    return False


# ============================================================================
# 4. 데이터셋 인덱스 구축 (크롭 샘플 생성)
# ============================================================================

def build_crop_index(
    data_dir: str,
    overlap_thresh: float = 0.3,
) -> List[Dict]:
    """
    GC10-DET 원본 → 크롭된 샘플 목록 생성

    각 원본 이미지 → Left 샘플 + Right 샘플 (총 2배)

    Returns: List of CropSample dict:
    {
        'src_img_path' : 원본 이미지 경로
        'src_xml_path' : 원본 XML 경로 (없으면 None)
        'crop_side'    : 'left' or 'right'
        'crop_xmin'    : 크롭 시작 x (원본 좌표)
        'crop_xmax'    : 크롭 끝 x
        'crop_w'       : 크롭 너비 (1024)
        'crop_h'       : 높이 (1000)
        'class_name'   : 결정된 클래스 (defect or 'good')
        'class_idx'    : 0~10
        'orig_folder'  : 원본 GC10 폴더 번호
        'objects'      : 크롭 기준 BBox 목록 (좌표 변환 완료)
        'stem'         : 저장 파일명 stem
    }
    """
    data_dir  = Path(data_dir)
    label_dir = data_dir / 'lable'
    if not label_dir.exists():
        label_dir = data_dir / 'label'
    if not label_dir.exists():
        label_dir = None
        print("[WARN] label 폴더 없음")

    xml_index = {}
    if label_dir:
        for f in label_dir.rglob('*.xml'):
            xml_index[f.stem] = str(f)
    print(f"XML {len(xml_index)}개 발견")

    IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_samples = []
    stats = defaultdict(lambda: defaultdict(int))  # stats[class_name][side]

    for folder_num in range(1, 11):
        folder_dir = data_dir / str(folder_num)
        if not folder_dir.exists():
            continue

        imgs = sorted([f for f in folder_dir.iterdir() if f.suffix.lower() in IMG_EXT])

        for img_path in imgs:
            xml_path = xml_index.get(img_path.stem)
            ann      = parse_voc_xml(xml_path) if xml_path else None

            img_w = ann['width']   if ann else 2048
            img_h = ann['height']  if ann else 1000
            half_w = img_w // 2

            objects = ann['objects'] if ann else []

            # BBox를 Left / Right 로 분리
            left_objs, right_objs = split_objects_by_crop(
                objects, img_w, img_h, overlap_thresh
            )

            for side, side_objs, x0, x1 in [
                ('left',  left_objs,  0,      half_w),
                ('right', right_objs, half_w, img_w ),
            ]:
                class_name, class_idx = determine_crop_class(side_objs)
                stem = f"{img_path.stem}_{side}"

                sample = {
                    'src_img_path': str(img_path),
                    'src_xml_path': xml_path,
                    'crop_side'   : side,
                    'crop_xmin'   : x0,
                    'crop_xmax'   : x1,
                    'crop_w'      : x1 - x0,
                    'crop_h'      : img_h,
                    'class_name'  : class_name,
                    'class_idx'   : class_idx,
                    'orig_folder' : str(folder_num),
                    'objects'     : side_objs,
                    'stem'        : stem,
                }
                all_samples.append(sample)
                stats[class_name][side] += 1

    # 통계 출력
    print(f"\n{'클래스':<20} {'left':>6} {'right':>6} {'합계':>6}")
    print("-" * 42)
    total = 0
    for cls in ALL_CLASSES:
        l = stats[cls]['left']
        r = stats[cls]['right']
        ko = GC10_CLASS_KO.get(
            next((k for k, v in GC10_CLASSES.items() if v == cls), 11), '정상'
        )
        print(f"  {cls:<18} {l:>6} {r:>6} {l+r:>6}  ({ko})")
        total += l + r
    print("-" * 42)
    print(f"  {'합계':<18} {total:>19}")
    print(f"\n원본 대비 {total / (total // 2):.1f}배 증가")
    return all_samples


# ============================================================================
# 5. Stratified Split
# ============================================================================

def stratified_split(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio:   float = 0.2,
    seed: int = 42,
) -> Tuple[List, List, List]:
    random.seed(seed)
    buckets = defaultdict(list)
    for s in samples:
        buckets[s['class_idx']].append(s)

    train, val, test = [], [], []
    for cls_idx in sorted(buckets):
        cls_s = buckets[cls_idx][:]
        random.shuffle(cls_s)
        n     = len(cls_s)
        n_tr  = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(cls_s[:n_tr])
        val.extend(cls_s[n_tr:n_tr + n_val])
        test.extend(cls_s[n_tr + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def print_split_stats(splits: Dict[str, List]):
    print("\n" + "=" * 70)
    print("  Split 통계 (클래스별)")
    print("=" * 70)
    split_names = list(splits.keys())
    header = f"  {'Class':<20}" + "".join([f" {s:>7}" for s in split_names]) + "   Total"
    print(header)
    print("  " + "-" * 65)

    counts = defaultdict(lambda: defaultdict(int))
    for sname, ssamples in splits.items():
        for s in ssamples:
            counts[s['class_name']][sname] += 1

    for cls in ALL_CLASSES:
        row   = f"  {cls:<20}"
        total = 0
        for sname in split_names:
            cnt   = counts[cls][sname]
            total += cnt
            row  += f" {cnt:>7}"
        row += f"   {total:>5}"
        print(row)

    print("  " + "-" * 65)
    totals_row = f"  {'TOTAL':<20}"
    grand = 0
    for sname in split_names:
        t = sum(counts[c][sname] for c in ALL_CLASSES)
        totals_row += f" {t:>7}"
        grand += t
    totals_row += f"   {grand:>5}"
    print(totals_row)
    print("=" * 70)


# ============================================================================
# 6-A. Classification 포맷 출력
# ============================================================================

def export_classification(
    splits: Dict[str, List],
    output_dir: str,
):
    """
    output_dir/classification/
    ├── train/
    │   ├── punching_hole/ ← 크롭된 이미지
    │   ├── welding_line/
    │   └── good/
    ├── val/
    └── test/
    """
    root = Path(output_dir) / 'classification'
    print(f"\n[Classification] 출력 → {root}")

    for split_name, samples in splits.items():
        for s in tqdm(samples, desc=f"  {split_name:5s}", leave=False):
            dst = root / split_name / s['class_name'] / f"{s['stem']}.jpg"
            crop_and_save(s['src_img_path'], s['crop_xmin'], s['crop_xmax'], dst)

    for sname, ssamples in splits.items():
        cls_counts = defaultdict(int)
        for s in ssamples:
            cls_counts[s['class_name']] += 1
        total = sum(cls_counts.values())
        good  = cls_counts.get('good', 0)
        print(f"  {sname:5s}: {total}장  (good={good}, defect={total-good})")

    print(f"  완료: {root}")
    return str(root)


# ============================================================================
# 6-B. YOLO 포맷 출력
# ============================================================================

def export_yolo(
    splits: Dict[str, List],
    output_dir: str,
):
    """
    output_dir/yolo/
    ├── images/ train/ val/ test/  ← 크롭 이미지
    ├── labels/ train/ val/ test/  ← .txt (YOLO 정규화 좌표)
    └── data.yaml

    Good 이미지: labels .txt 파일을 빈 파일로 저장 (YOLO background)
    """
    root = Path(output_dir) / 'yolo'
    print(f"\n[YOLO] 출력 → {root}")

    for split_name, samples in splits.items():
        img_dir = root / 'images' / split_name
        lbl_dir = root / 'labels' / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for s in tqdm(samples, desc=f"  {split_name:5s}", leave=False):
            img_dst = img_dir / f"{s['stem']}.jpg"
            lbl_dst = lbl_dir / f"{s['stem']}.txt"

            # 이미지 크롭 저장
            crop_and_save(s['src_img_path'], s['crop_xmin'], s['crop_xmax'], img_dst)

            # YOLO 라벨 생성
            cw = s['crop_w']   # 1024
            ch = s['crop_h']   # 1000

            valid_objs = [o for o in s['objects'] if not o['difficult']]

            with open(lbl_dst, 'w') as f:
                if not valid_objs:
                    # Good → 빈 파일 (background)
                    pass
                else:
                    for obj in valid_objs:
                        cx = (obj['xmin'] + obj['xmax']) / 2 / cw
                        cy = (obj['ymin'] + obj['ymax']) / 2 / ch
                        bw = (obj['xmax'] - obj['xmin']) / cw
                        bh = (obj['ymax'] - obj['ymin']) / ch

                        # 범위 클램핑
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        bw = max(0.001, min(1.0, bw))
                        bh = max(0.001, min(1.0, bh))

                        f.write(f"{obj['class_idx']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # data.yaml (good 포함 11클래스)
    yaml_path = root / 'data.yaml'
    lines = [
        f"path: {root.resolve()}",
        f"train: images/train",
        f"val:   images/val",
        f"test:  images/test",
        f"",
        f"nc: {len(ALL_CLASSES)}",
        f"names:",
    ]
    for i, name in enumerate(ALL_CLASSES):
        ko = GC10_CLASS_KO.get(i + 1, '정상') if i < 10 else '정상'
        lines.append(f"  {i}: {name}  # {ko}")

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    for sname, ssamples in splits.items():
        print(f"  {sname:5s}: {len(ssamples)}장")
    print(f"  data.yaml: {yaml_path}")
    print(f"  완료: {root}")
    return str(root)


# ============================================================================
# 6-C. VOC 포맷 출력
# ============================================================================

def make_voc_xml(
    stem: str,
    class_name: str,
    crop_w: int,
    crop_h: int,
    objects: List[Dict],
) -> str:
    """크롭된 이미지용 Pascal VOC XML 문자열 생성"""
    lines = [
        '<annotation>',
        f'\t<folder>{class_name}</folder>',
        f'\t<filename>{stem}.jpg</filename>',
        '\t<source><database>GC10-DET-CROP</database></source>',
        '\t<size>',
        f'\t\t<width>{crop_w}</width>',
        f'\t\t<height>{crop_h}</height>',
        '\t\t<depth>3</depth>',
        '\t</size>',
        '\t<segmented>0</segmented>',
    ]

    for obj in objects:
        if obj['difficult']:
            continue
        lines += [
            '\t<object>',
            f'\t\t<name>{obj["class_name"]}</name>',
            '\t\t<pose>Unspecified</pose>',
            '\t\t<truncated>0</truncated>',
            '\t\t<difficult>0</difficult>',
            '\t\t<bndbox>',
            f'\t\t\t<xmin>{obj["xmin"]}</xmin>',
            f'\t\t\t<ymin>{obj["ymin"]}</ymin>',
            f'\t\t\t<xmax>{obj["xmax"]}</xmax>',
            f'\t\t\t<ymax>{obj["ymax"]}</ymax>',
            '\t\t</bndbox>',
            '\t</object>',
        ]

    lines.append('</annotation>')
    return '\n'.join(lines)


def export_voc(
    splits: Dict[str, List],
    output_dir: str,
):
    """
    output_dir/voc/
    ├── train/
    │   ├── images/       ← 크롭 이미지
    │   └── annotations/  ← 새로 생성된 XML (크롭 좌표 기준)
    ├── val/
    └── test/
    """
    root = Path(output_dir) / 'voc'
    print(f"\n[VOC] 출력 → {root}")

    for split_name, samples in splits.items():
        img_dir = root / split_name / 'images'
        ann_dir = root / split_name / 'annotations'
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        for s in tqdm(samples, desc=f"  {split_name:5s}", leave=False):
            # 이미지 크롭
            img_dst = img_dir / f"{s['stem']}.jpg"
            crop_and_save(s['src_img_path'], s['crop_xmin'], s['crop_xmax'], img_dst)

            # XML 생성 (크롭 기준 좌표)
            xml_str = make_voc_xml(
                stem       = s['stem'],
                class_name = s['class_name'],
                crop_w     = s['crop_w'],
                crop_h     = s['crop_h'],
                objects    = s['objects'],
            )
            xml_dst = ann_dir / f"{s['stem']}.xml"
            with open(xml_dst, 'w', encoding='utf-8') as f:
                f.write(xml_str)

    for sname, ssamples in splits.items():
        print(f"  {sname:5s}: {len(ssamples)}장")
    print(f"  완료: {root}")
    return str(root)


# ============================================================================
# 7. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='GC10-DET Horizontal Crop Splitter')

    parser.add_argument('--data_dir',       type=str, required=True,
                        help='GC10-DET 루트 경로')
    parser.add_argument('--output_dir',     type=str, default='./gc10_cropped',
                        help='출력 루트 경로')
    parser.add_argument('--format',         type=str, default='all',
                        choices=['all', 'cls', 'yolo', 'voc'],
                        help='출력 포맷')
    parser.add_argument('--train',          type=float, default=0.8)
    parser.add_argument('--val',            type=float, default=0.2)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--overlap_thresh', type=float, default=0.3,
                        help='BBox 면적의 몇 %% 이상 겹쳐야 해당 절반에 포함 (default 0.3)')
    parser.add_argument('--save_json',      action='store_true',
                        help='split index JSON 저장')

    args = parser.parse_args()

    test_ratio = round(1.0 - args.train - args.val, 4)
    assert test_ratio > 0, "train + val 합이 1.0 이상"

    print("=" * 70)
    print("  GC10-DET Horizontal Crop Splitter")
    print("=" * 70)
    print(f"  data_dir       : {args.data_dir}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  분할 비율       : train={args.train} / val={args.val} / test={test_ratio}")
    print(f"  overlap_thresh : {args.overlap_thresh}  "
          f"(bbox 면적의 {args.overlap_thresh*100:.0f}% 이상 겹쳐야 포함)")
    print(f"  포맷           : {args.format}")
    print()

    # 크롭 샘플 인덱스 구축
    print("크롭 샘플 인덱스 구축 중...")
    samples = build_crop_index(args.data_dir, args.overlap_thresh)

    # Train/Val/Test 분할
    print("\nStratified Split 수행...")
    train_s, val_s, test_s = stratified_split(samples, args.train, args.val, args.seed)
    splits = {'train': train_s, 'val': val_s, 'test': test_s}

    print_split_stats(splits)

    # JSON 저장 (옵션)
    if args.save_json:
        jdir = Path(args.output_dir) / 'splits_json'
        jdir.mkdir(parents=True, exist_ok=True)
        for name, data in splits.items():
            with open(jdir / f'{name}.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON 저장: {jdir}")

    # 포맷 출력
    print("\n파일 저장 중 (이미지 크롭 + 라벨 생성)...")
    exported = {}

    if args.format in ('all', 'cls'):
        exported['classification'] = export_classification(splits, args.output_dir)

    if args.format in ('all', 'yolo'):
        exported['yolo'] = export_yolo(splits, args.output_dir)

    if args.format in ('all', 'voc'):
        exported['voc'] = export_voc(splits, args.output_dir)

    # 최종 요약
    print("\n" + "=" * 70)
    print("  완료! 생성된 경로:")
    for fmt, path in exported.items():
        print(f"    [{fmt:>14}] {path}")
    print()
    print("  YOLO 학습 예시:")
    if 'yolo' in exported:
        print(f"    yolo train data={exported['yolo']}/data.yaml model=yolov8n.pt imgsz=1024")
    print()
    print("  SAAD-Net / Classification 학습 예시:")
    if 'classification' in exported:
        print(f"    python train_unified.py --data_dir {exported['classification']} "
              f"--model defect_lock_v2_imp --num_classes 11")
    print("=" * 70)


# ============================================================================
# 8. 직접 실행 도움말
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        print("GC10-DET Horizontal Crop Splitter")
        print()
        print("사용법:")
        print("  python gc10_crop_split.py --data_dir /path/to/GC10-DET --output_dir ./gc10_cropped")
        print()
        print("옵션:")
        print("  --format yolo          YOLO만 출력")
        print("  --format cls           Classification만 출력")
        print("  --format voc           VOC만 출력")
        print("  --train 0.8 --val 0.2  비율 조정")
        print("  --overlap_thresh 0.1   BBox 10% 이상 겹치면 포함 (더 많은 defect 포함)")
        print("  --save_json            split index JSON도 저장")
        print()
        print("생성 구조:")
        print("  gc10_cropped/")
        print("  ├── classification/")
        print("  │   ├── train/ val/ test/")
        print("  │   └── {class_name}/ ← 크롭 이미지 (1024×1000)")
        print("  │       └── good/     ← defect 없는 절반 자동 생성")
        print("  ├── yolo/")
        print("  │   ├── images/ train/ val/ test/")
        print("  │   ├── labels/ train/ val/ test/ ← YOLO .txt")
        print("  │   └── data.yaml  ← nc=11 (10 defect + good)")
        print("  └── voc/")
        print("      ├── train/ val/ test/")
        print("      └── images/ + annotations/ ← 크롭 좌표 기준 새 XML")
        print()
        print("분할 로직:")
        print("  원본 2048×1000  →  LEFT 1024×1000  +  RIGHT 1024×1000")
        print("  BBox가 해당 절반에 30% 이상 겹치면 포함 (--overlap_thresh로 조정)")
        print("  BBox 없는 절반 → good 클래스 자동 배정")
        print("  BBox 여러 종 혼재 → 면적 가장 큰 클래스로 단일 배정")
    else:
        main()
