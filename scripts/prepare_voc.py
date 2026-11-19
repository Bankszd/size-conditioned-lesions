#!/usr/bin/env python3
"""Convert a PASCAL VOC detection split to COCO-format annotations.

Two conventions matter and are used by every VOC number in the paper:

* the 20 VOC classes are mapped onto their COCO category ids, so that the COCO
  evaluator can be used unchanged (mapping table in ``lesions/data``);
* a ``difficult`` object becomes ``iscrowd = 1``, which makes the official COCO
  evaluator ignore it exactly like a crowd region.  Non-difficult objects are
  ordinary annotations.

    python scripts/prepare_voc.py --voc-root /path/to/VOCdevkit/VOC2007 \
        --split test --out data/voc2007test

``VOC2012`` with ``--split val`` does the same for the 5,823-image validation
split used as the second confirmation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

sys_path = None
try:  # the class map lives in the package, but keep this script standalone
    from lesions.data import VOC_TO_COCO_IDS, VOC_CLASSES
except Exception:  # pragma: no cover - fallback when run outside the repo
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]
    VOC_TO_COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                       44, 62, 63, 64, 67, 72]

CLASS_TO_COCO = dict(zip(VOC_CLASSES, VOC_TO_COCO_IDS))


def image_id(stem):
    """Year-prefixed VOC ids (2008_000123) become unique integers."""
    return int(stem.replace("_", ""))


def convert(voc_root, split):
    voc_root = Path(voc_root)
    id_file = voc_root / "ImageSets" / "Main" / ("%s.txt" % split)
    stems = [line.strip() for line in id_file.read_text().splitlines() if line.strip()]
    stems.sort()

    images, annotations = [], []
    ann_id = 1
    for stem in stems:
        xml_path = voc_root / "Annotations" / ("%s.xml" % stem)
        jpeg_path = voc_root / "JPEGImages" / ("%s.jpg" % stem)
        if not xml_path.exists() or not jpeg_path.exists():
            raise SystemExit("missing file for %s" % stem)
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        image_id_ = image_id(stem)
        images.append({"id": image_id_, "file_name": jpeg_path.name,
                       "width": width, "height": height})
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_TO_COCO:
                continue
            box = obj.find("bndbox")
            x0 = float(box.find("xmin").text) - 1.0
            y0 = float(box.find("ymin").text) - 1.0
            x1 = float(box.find("xmax").text) - 1.0
            y1 = float(box.find("ymax").text) - 1.0
            w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
            difficult = int(obj.find("difficult").text)
            annotations.append({
                "id": ann_id,
                "image_id": image_id_,
                "category_id": CLASS_TO_COCO[name],
                "bbox": [x0, y0, w, h],
                "area": w * h,
                "iscrowd": 1 if difficult else 0,
            })
            ann_id += 1
    categories = [{"id": cid, "name": name, "supercategory": "voc"}
                  for name, cid in CLASS_TO_COCO.items()]
    return {"info": {"description": "VOC %s converted to COCO format" % split},
            "images": images, "annotations": annotations,
            "categories": categories}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/voc")
    args = ap.parse_args()

    payload = convert(args.voc_root, args.split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / ("instances_%s.json" % args.split)
    text = json.dumps(payload)
    target.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    n_diff = sum(1 for a in payload["annotations"] if a["iscrowd"])
    print("images %d, annotations %d (%d difficult -> iscrowd)"
          % (len(payload["images"]), len(payload["annotations"]), n_diff))
    print("wrote %s" % target)
    print("content sha256 %s" % digest)


if __name__ == "__main__":
    main()
