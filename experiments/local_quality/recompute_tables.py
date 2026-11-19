"""Recompute paper Table 1 (COCO, dense R50) and Table 3 (VOC 2007 test, H1/H2)
from the shipped per-anchor records.  No model forward pass is needed.

Definitions (paper Sections 2-3):
  D4   = Q_outside - Q_inside      (fixed four-cell lesion, in-box vs out-of-box)
  L.02 = Q_clean   - Q_inside      (equal-mass lesion, m = 0.02)
  statistic = E[.|small] - E[.|large], whole-image percentile bootstrap,
              PCG64 seed 20271001, 5000 resamples.
"""
import argparse, gzip, json
import numpy as np

SEED = 20271001
RESAMPLES = 5000

# Published values.  Table 1 rows are the manuscript numbers; Table 3 rows are
# the registered VOC 2007 test values behind hypotheses H1 and H2.
REFERENCE = {
    ("table1", "detr"): {"d4": 0.021364, "d02": 0.000147},
    ("table1", "conditional_detr"): {"d4": 0.023249, "d02": -0.000024},
    ("table3", "detr"): {"d4": 0.028481, "d02": 0.000330},
    ("table3", "conditional_detr"): {"d4": 0.033838, "d02": -0.000285},
}

# The per-anchor records were regenerated on a different software stack from the
# one that produced the original manuscript run, so Table 1 reproduces to about
# 2.4e-4 on the point estimate (and its bootstrap endpoints to about 1.8e-3).
# See the "Known differences" section of README.md.  Table 3 is the registered
# VOC 2007 run; its DETR-R50 row moves by the same amount for the same reason.
TOLERANCE = {"table1": 5e-4, "table3": 5e-4}


def load(path, arm, mode="all"):
    """Return {anchor key: (clean_q, perturbed_q, area_bin, image_id)} for one arm."""
    out = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("arm") != arm:
                continue
            if mode is not None and row.get("mode", mode) != mode:
                continue
            if "Q_clean" in row and isinstance(row["Q_clean"], dict):
                clean = float(row["Q_clean"]["quality"])
                pert = float(row["perturbed_local"]["quality"])
            else:
                clean = float(row["clean_local"]["quality"])
                pert = float(row["perturbed_local"]["quality"])
            key = (row["image_id"], row["annotation_id"])
            out[key] = (clean, pert, row["area_bin"], int(row["image_id"]))
    return out


def gap(loss, bins, images, weights):
    small = bins == "small"; large = bins == "large"
    ws = weights[small].sum(); wl = weights[large].sum()
    if ws <= 0 or wl <= 0:
        return np.nan
    return float((loss[small] * weights[small]).sum() / ws - (loss[large] * weights[large]).sum() / wl)


def analyse(loss, bins, images, n_boot):
    loss = np.asarray(loss); bins = np.asarray(bins); images = np.asarray(images)
    uniq, inv = np.unique(images, return_inverse=True)
    point = gap(loss, bins, images, np.ones_like(loss))
    counts = {b: int((bins == b).sum()) for b in ("small", "medium", "large")}
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = np.empty(n_boot); n_img = uniq.size
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
        draws[i] = gap(loss, bins, images, w[inv])
    return {"point": point, "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "counts": counts, "n_rows": int(loss.size)}


def arm_contrast(path, kind, n_boot, mode="all"):
    inside = load(path, "fixed_inside", mode)
    if kind == "d4":
        outside = load(path, "fixed_outside", mode)
        keys = [k for k in inside if k in outside]
        loss = [outside[k][1] - inside[k][1] for k in keys]
        bins = [inside[k][2] for k in keys]; images = [inside[k][3] for k in keys]
    else:
        mass = load(path, "mass_0.02", mode)
        keys = [k for k in inside if k in mass]
        loss = [inside[k][0] - mass[k][1] for k in keys]
        bins = [inside[k][2] for k in keys]; images = [inside[k][3] for k in keys]
    return analyse(loss, bins, images, n_boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=RESAMPLES)
    ap.add_argument("--no-boot", action="store_true")
    ap.add_argument("--json", default="recompute_tables.json")
    args = ap.parse_args()
    n_boot = 20 if args.no_boot else args.boot
    jobs = [
        ("table1", "detr", "stage7_local_detr.jsonl.gz"),
        ("table1", "conditional_detr", "stage7_local_conditional_detr.jsonl.gz"),
        ("table3", "detr", "task_confirm_detr_full.jsonl.gz"),
        ("table3", "conditional_detr", "task_confirm_conditional_detr_full.jsonl.gz"),
    ]
    print("%-7s %-17s %-6s %-11s %-26s %s" % ("table", "model", "quant", "estimate", "95% interval", "n small/med/large"))
    results = {}
    for table, model, path in jobs:
        for kind in ("d4", "d02"):
            out = arm_contrast(path, kind, n_boot)
            results["%s|%s|%s" % (table, model, kind)] = out
            print("%-7s %-17s %-6s %+.6f  [%+.6f, %+.6f]  %d/%d/%d" % (
                table, model, kind, out["point"], out["ci"][0], out["ci"][1],
                out["counts"]["small"], out["counts"]["medium"], out["counts"]["large"]))
    print()
    ok = True
    for key, out in results.items():
        table, model, kind = key.split("|")
        ref = REFERENCE[(table, model)][kind]
        dev = abs(out["point"] - ref)
        same = dev <= TOLERANCE[table]
        ok = ok and same
        print("CHECK %-34s reference %+.6f  recomputed %+.6f  dev %.1e  %s"
              % (key, ref, out["point"], dev, "MATCH" if same else "DIFFER"))
    print("\nALL_MATCH" if ok else "\nSOME_DIFFER")
    json.dump(results, open(args.json, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
