#!/usr/bin/env python3
"""Download the seven checkpoints listed in configs/models.yaml and verify them.

    python scripts/download_checkpoints.py --ckpt-root weights

Nothing is downloaded if ``--check-only`` is passed, which is the mode used by
``docs/REPRODUCE.md`` when the weights are already on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="weights")
    ap.add_argument("--config", default=str(ROOT / "configs/models.yaml"))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())["models"]
    root = Path(args.ckpt_root)
    failures = 0
    for key, entry in spec.items():
        dest = root / entry["file"]
        if not dest.exists():
            if args.check_only:
                print("%-24s MISSING  %s" % (key, dest))
                failures += 1
                continue
            print("%-24s downloading %s" % (key, entry["url"]))
            try:
                fetch(entry["url"], dest)
            except Exception as exc:  # network problems are reported, not hidden
                print("%-24s DOWNLOAD FAILED: %s" % (key, exc))
                failures += 1
                continue
        got = sha256(dest)
        ok = got == entry["sha256"]
        print("%-24s %s  %s" % (key, got, "OK" if ok else
                                "SHA256 MISMATCH (expected %s)" % entry["sha256"]))
        failures += 0 if ok else 1
    print("\n%d problem(s)" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
