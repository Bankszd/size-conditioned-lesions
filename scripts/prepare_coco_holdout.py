#!/usr/bin/env python3
"""Rebuild the frozen COCO val2017 holdout split.

5,000 val2017 image ids are permuted with NumPy PCG64 seed 20270925; positions
1000-4999 are the 4,000-image holdout used by every COCO analysis.  Positions
0-999 were used for development during the original work and are never used as
a confirmation set in this revision.

    python scripts/prepare_coco_holdout.py --annotations instances_val2017.json \
        --out data/coco_holdout.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 20270925
SKIP = 1000
KEEP = 4000


def holdout_ids(annotation_file):
    ann = json.loads(Path(annotation_file).read_text())
    ids = sorted(int(image["id"]) for image in ann["images"])
    if len(ids) != 5000:
        raise SystemExit("expected 5,000 val2017 images, found %d" % len(ids))
    perm = np.random.Generator(np.random.PCG64(SEED)).permutation(ids)
    return [int(x) for x in perm[SKIP:SKIP + KEEP]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="data/coco_holdout.json")
    args = ap.parse_args()

    ids = holdout_ids(args.annotations)
    payload = {"seed": SEED, "skip": SKIP, "keep": KEEP, "image_ids": ids}
    text = json.dumps(payload, indent=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("wrote %s (%d ids)" % (out, len(ids)))
    print("content sha256 %s" % digest)
    print("first five ids %s" % ids[:5])


if __name__ == "__main__":
    main()
