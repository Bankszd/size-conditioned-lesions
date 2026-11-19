#!/usr/bin/env python3
"""Rebuild every table of the paper from the shipped records.

CPU only, no GPU, no checkpoint, no image file: everything is recomputed from
the per-anchor records and the bootstrap arrays under ``records/``.

Run ``python scripts/reproduce_tables.py --out results/recomputed`` or, more
simply, ``make tables``.

The heavy numerical work is done by the frozen scripts under ``experiments/``;
this file only invokes them in the right order, with the right inputs, and
collects their outputs.  Nothing here re-implements a statistic.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

T_LINE = re.compile(r"^(T\s*= I_four - I_prop50)\s+([+-][\d.]+)\s+"
                    r"\[([+-][\d.]+),\s*([+-][\d.]+)\]\s+pp", re.M)


def run(cmd, cwd, stdout_file=None):
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit("FAILED: %s" % " ".join(cmd))
    text = proc.stdout
    if stdout_file is not None:
        Path(stdout_file).write_text(text, encoding="utf-8")
    print(text.rstrip())
    print()
    return text


def section(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


def table1_table3(out):
    section("Table 1 (COCO holdout) and Table 3 H1/H2 (VOC 2007 test): "
            "fixed four cells and equal mass 0.02")
    target = out / "table1_table3.json"
    run([PY, ROOT / "experiments/local_quality/recompute_tables.py",
         "--json", target],
        cwd=ROOT / "records/local_quality")
    return json.loads(target.read_text())


def model_comparison(out):
    section("Table Y1 / Table 3 model comparison: D, I and T on three datasets")
    rec = ROOT / "records/model_comparison"
    runs = {
        "coco": ("task_y_boot_geo_detr.npz",
                 "task_y_boot_geo_conditional_detr.npz"),
        "voc2007": ("task_y_boot_yv_detr.npz",
                    "task_y_boot_yv_conditional_detr.npz"),
        "voc2012": ("task_y_boot_y2_detr.npz",
                    "task_y_boot_y2_cond.npz"),
    }
    got = {}
    for tag, (det, cond) in runs.items():
        text = run([PY, ROOT / "experiments/model_comparison/recompute_y.py",
                    "--npz", rec / det, rec / cond],
                   cwd=ROOT, stdout_file=out / ("model_comparison_%s.txt" % tag))
        m = T_LINE.search(text)
        if not m:
            raise SystemExit("could not parse T for %s" % tag)
        got[tag] = {"T": float(m.group(2)),
                    "ci": [float(m.group(3)), float(m.group(4))]}
    return got


def relative_dose(out):
    section("Table R: relative dose versus absolute dose on VOC 2007 test")
    rec = ROOT / "records/relative_dose"
    target = out / "tableR.json"
    run([PY, ROOT / "experiments/relative_dose/r_voc_analyze.py",
         rec / "r_voc_detr.jsonl.gz", rec / "r_voc_conditional_detr.jsonl.gz",
         target], cwd=ROOT)
    return json.loads(target.read_text())


def fixed_area(out):
    section("Table N: fixed-area lesions on VOC 2007 test (DEIMv2-S records)")
    rec = ROOT / "records/deformable"
    target = out / "tableN.json"
    run([PY, ROOT / "experiments/fixed_area/n_voc_analyze.py",
         "--deim", rec / "task_d2_n2_deim_voc_officialcfg.jsonl.gz",
         "--out", target], cwd=ROOT)
    return json.loads(target.read_text())


def transfer(out):
    section("Table X: frozen COCO dose-effect curves applied to VOC 2007 test")
    got = {}
    for model, coco, voc in (
        ("detr", "task_m_detr.jsonl.gz", "r_voc_detr.jsonl.gz"),
        ("conditional_detr", "task_m_conditional_detr.jsonl.gz",
         "r_voc_conditional_detr.jsonl.gz"),
    ):
        target = out / ("tableX_%s.json" % model)
        run([PY, ROOT / "experiments/transfer/task_x.py",
             "--model", model,
             "--coco", ROOT / "records/transfer" / coco,
             "--voc", ROOT / "records/relative_dose" / voc,
             "--out", target], cwd=ROOT)
        got[model] = json.loads(target.read_text())
    return got


def shipped_summary(out):
    """Print the blocks whose inputs are per-image detector outputs.

    Those outputs are far larger than the release budget, so the release ships
    the aggregate result files produced by the frozen scripts instead of the
    predictions themselves.  See docs/EXPECTED.md.
    """
    section("Shipped aggregates (per-image detector outputs are not included)")
    files = ["tableH3_voc.json", "tableH4_voc.json", "tableD_voc.json",
             "tableG_voc.json", "tableG5_voc.json", "tableN_voc.json"]
    payload = {}
    for name in files:
        path = ROOT / "results" / name
        if not path.exists():
            print("missing: results/%s" % name)
            continue
        payload[name] = json.loads(path.read_text())
        print("results/%s" % name)
        print(json.dumps(payload[name], indent=1, sort_keys=True)[:4000])
        print()
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/recomputed")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "table1_table3": table1_table3(out),
        "model_comparison": model_comparison(out),
        "tableR": relative_dose(out),
        "tableN": fixed_area(out),
        "tableX": transfer(out),
        "shipped": shipped_summary(out),
    }
    (out / "all_tables.json").write_text(
        json.dumps(results, indent=1, sort_keys=True, default=str),
        encoding="utf-8")
    print("wrote %s" % (out / "all_tables.json"))


if __name__ == "__main__":
    main()
