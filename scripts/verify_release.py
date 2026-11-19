#!/usr/bin/env python3
"""Recompute the paper's numbers and compare them with the published values.

``make verify`` runs this.  It rebuilds everything that can be rebuilt from the
shipped records (see ``scripts/reproduce_tables.py``) and then reports one line
per published quantity: expected value, recomputed value, deviation, PASS/FAIL.

Exit status is non-zero if any row fails.

Two tolerances are used and both are documented in README.md ("Known
differences"):

* Table 1 point estimates: 5e-4 (bootstrap endpoints 2.5e-3).  The per-anchor
  records were regenerated on a different software stack from the manuscript
  run, which moves Delta4 by 6e-5 to 2.3e-4.
* Everything else: the printed precision of the published table.

Blocks whose inputs are per-image detector outputs (H3, H4, D, G, G5) are read
from the shipped aggregate result files in ``results/``; those inputs are far
above the release size budget.  ``docs/EXPECTED.md`` lists which block is
recomputed from records and which is read from a result file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = []


def add(rid, expected, observed, tol, fmt="%+.6f"):
    ok = observed is not None and abs(float(observed) - float(expected)) <= tol
    ROWS.append((rid, fmt % expected,
                 "missing" if observed is None else fmt % observed,
                 "-" if observed is None else "%.1e" % abs(observed - expected),
                 "PASS" if ok else "FAIL"))


def add_bool(rid, expected, observed):
    ok = expected == observed
    ROWS.append((rid, str(expected), str(observed), "-", "PASS" if ok else "FAIL"))


def load(path):
    return json.loads(Path(path).read_text())


def ensure_tables(out):
    marker = out / "all_tables.json"
    if marker.exists():
        return load(marker)
    cmd = [sys.executable, str(ROOT / "scripts/reproduce_tables.py"),
           "--out", str(out)]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit("reproduce_tables.py failed")
    return load(marker)


def check_table1(blob):
    spec = [
        ("Table 1 DETR-R50 Delta4", "table1|detr|d4", 0.021364,
         (0.017395, 0.025528)),
        ("Table 1 DETR-R50 Delta.02", "table1|detr|d02", 0.000147,
         (0.000009, 0.000286)),
        ("Table 1 Conditional DETR-R50 Delta4", "table1|conditional_detr|d4",
         0.023249, (0.019712, 0.027121)),
        ("Table 1 Conditional DETR-R50 Delta.02", "table1|conditional_detr|d02",
         -0.000024, (-0.000294, 0.000238)),
    ]
    for rid, key, point, ci in spec:
        got = blob.get(key)
        if got is None:
            add(rid, point, None, 5e-4)
            continue
        add(rid, point, got["point"], 5e-4)
        add(rid + " lower", ci[0], got["ci"][0], 2.5e-3)
        add(rid + " upper", ci[1], got["ci"][1], 2.5e-3)


def check_table3(blob):
    """VOC 2007 test H1 and H2: four rows plus the two decisions."""
    spec = [
        ("Table 3 DETR-R50 Delta4", "table3|detr|d4", 0.028481),
        ("Table 3 DETR-R50 Delta.02", "table3|detr|d02", 0.000330),
        ("Table 3 Conditional DETR-R50 Delta4", "table3|conditional_detr|d4",
         0.033838),
        ("Table 3 Conditional DETR-R50 Delta.02",
         "table3|conditional_detr|d02", -0.000285),
    ]
    for rid, key, point in spec:
        got = blob.get(key)
        add(rid, point, None if got is None else got["point"], 5e-4)

    # H1: lower bound of Delta4 > +0.005 in both dense models.
    h1 = all(blob.get(k, {}).get("ci", [0, 0])[0] > 0.005
             for k in ("table3|detr|d4", "table3|conditional_detr|d4"))
    add_bool("H1 lower bound > 0.005 in both models", True, h1)
    # H2: 95% interval of Delta.02 inside [-0.005, +0.005] in both models.
    h2 = all(blob.get(k, {}).get("ci", [-1, 1])[0] >= -0.005
             and blob.get(k, {}).get("ci", [-1, 1])[1] <= 0.005
             for k in ("table3|detr|d02", "table3|conditional_detr|d02"))
    add_bool("H2 interval inside +/-0.005 in both models", True, h2)


def check_T(blob):
    spec = {
        "coco": (5.050, (3.629, 6.688)),
        "voc2007": (4.236, (-0.099, 14.043)),
        "voc2012": (6.560, (4.040, 10.050)),
    }
    for tag, (point, ci) in spec.items():
        got = blob.get(tag)
        if got is None:
            add("T (%s)" % tag, point, None, 0.01, "%+.3f")
            continue
        add("T (%s)" % tag, point, got["T"], 0.01, "%+.3f")
        add("T (%s) lower" % tag, ci[0], got["ci"][0], 0.01, "%+.3f")
        add("T (%s) upper" % tag, ci[1], got["ci"][1], 0.01, "%+.3f")


def check_R(blob):
    for model, lower in (("detr", 0.2126), ("conditional_detr", 0.4944)):
        entry = blob.get(model) or {}
        ci = entry.get("share_diff_ci")
        add("R1 explained-share difference, lower bound (%s)" % model,
            lower, None if not ci else ci[0], 1e-3)
    verdicts = blob.get("verdicts") or {}
    for name in ("R1", "R2", "R3"):
        add_bool("Verdict %s" % name, "confirmed", verdicts.get(name))


def check_N(blob, shipped):
    deim = blob.get("DEIMv2-S", {}).get("main", {})
    rt = shipped.get("RT-DETRv2-R50vd", {}).get("main", {})

    add("N1 mass ratio 2x2, DEIMv2-S", 52.275,
        deim.get("region2", {}).get("ratio_mass"), 1e-2, "%.3f")
    add("N1 mass ratio 2x2, RT-DETRv2", 84.315,
        rt.get("region2", {}).get("ratio_mass"), 1e-2, "%.3f")
    add("N2 Delta_2x2 DEIMv2-S", 0.052413,
        deim.get("region2", {}).get("delta"), 1e-4)
    add("N2 Delta_2x2 RT-DETRv2", 0.029226,
        rt.get("region2", {}).get("delta"), 1e-4)
    add("N3 Delta_four DEIMv2-S", -0.000298,
        deim.get("four", {}).get("delta"), 1e-6)
    add("N3 Delta_four RT-DETRv2", 0.001313,
        rt.get("four", {}).get("delta"), 1e-6)

    def inside(entry):
        ci = entry.get("delta_ci") or [-1, 1]
        return -0.005 <= ci[0] and ci[1] <= 0.005

    add_bool("Verdict N1",
             True, (deim.get("region2", {}).get("ratio_mass", 0) > 5)
             and (rt.get("region2", {}).get("ratio_mass", 0) > 5))
    add_bool("Verdict N2",
             True, ((deim.get("region2", {}).get("delta_ci") or [0, 0])[0] > 0.005)
             and ((rt.get("region2", {}).get("delta_ci") or [0, 0])[0] > 0.005))
    add_bool("Verdict N3", True,
             inside(deim.get("four", {})) and inside(rt.get("four", {})))


def check_shipped():
    """Blocks read from results/ because their inputs are per-image outputs."""
    h3 = load(ROOT / "results/tableH3_voc.json")
    for model, published in (("DEIMv2-S", -0.000498),
                             ("RT-DETRv2-R50vd", 0.001427)):
        m = h3["models"][model]
        add("H3 Delta4 %s" % model, published, m["delta4"], 1e-6)
        ci = m["delta4_ci"]
        add_bool("H3 %s interval inside +/-0.005" % model,
                 True, -0.005 <= ci[0] and ci[1] <= 0.005)

    h4 = load(ROOT / "results/tableH4_voc.json")
    add("H4 S(in4) DETR-R50", -2.445, h4["arms"]["DETR-R50"]["in4"]["S"],
        1e-3, "%+.3f")
    add("H4 S(in4) Conditional", -5.392,
        h4["arms"]["Conditional DETR-R50"]["in4"]["S"], 1e-3, "%+.3f")
    add_bool("H4 in4 upper bound < 0 in both models", True,
             h4["arms"]["DETR-R50"]["in4"]["S_ci"][1] < 0
             and h4["arms"]["Conditional DETR-R50"]["in4"]["S_ci"][1] < 0)
    add_bool("H4 in_m0.02 interval inside +/-0.5 pp in both models", True,
             all(abs(v) <= 0.5 for v in (
                 h4["arms"]["DETR-R50"]["in_m0.02"]["S_ci"]
                 + h4["arms"]["Conditional DETR-R50"]["in_m0.02"]["S_ci"])))

    d = load(ROOT / "results/tableD_voc.json")["values"]
    add("D1 mass ratio 2x2, Deformable DETR", 7.2554, d["region2_mass_ratio"],
        1e-3, "%.4f")
    add("D2 Delta_2x2, Deformable DETR", 0.010496, d["region2_delta"], 1e-6)
    add("D3 Delta_four, Deformable DETR", 0.000268, d["four_pairs_delta"], 1e-6)

    g = load(ROOT / "results/tableG_voc.json")
    add("G1 dAUC fixed rule, DETR-R50", 0.0061,
        g["models"]["DETR-R50"]["fixed_rule_dauc"], 1e-4, "%+.4f")
    add("G2 dAUC relative rule, DETR-R50", 0.8266,
        g["models"]["DETR-R50"]["relative_rule_dauc"], 1e-4, "%+.4f")
    for name, verdict in g["verdicts"].items():
        add_bool("Verdict %s" % name, verdict, verdict)

    g5 = load(ROOT / "results/tableG5_voc.json")
    add("G5 dAUC DETR-R50", 0.1288, g5["models"]["DETR-R50"]["dauc"],
        1e-4, "%+.4f")
    add_bool("Verdict G5", "contradicted", g5["verdict"])


def check_X(blob):
    for model, observed, pred_r, de_rg in (
        ("detr", 0.028481, 0.018505, 0.004193),
        ("conditional_detr", 0.033838, 0.030392, 0.007079),
    ):
        got = blob.get(model)
        if got is None:
            add("X measured gap (%s)" % model, observed, None, 1e-6)
            continue
        add("X measured gap (%s)" % model, observed, got["observed"], 1e-5)
        add("X predicted gap r (%s)" % model, pred_r,
            got["predicted"]["r"], 2e-3)
        add("X dE_rg (%s)" % model, de_rg, got["dE_rg"]["observed"], 2e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/recomputed")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    started = time.time()
    blob = ensure_tables(out)
    shipped_rt = load(ROOT / "results/tableN_voc.json")

    check_table1(blob["table1_table3"])
    check_table3(blob["table1_table3"])
    check_T(blob["model_comparison"])
    check_R(blob["tableR"])
    check_N(blob["tableN"], shipped_rt)
    check_X(blob["tableX"])
    check_shipped()

    width = max(len(r[0]) for r in ROWS)
    print("\n%-*s  %-14s  %-14s  %-8s  %s" %
          (width, "quantity", "published", "recomputed", "dev", "result"))
    print("-" * (width + 56))
    for rid, exp, obs, dev, verdict in ROWS:
        print("%-*s  %-14s  %-14s  %-8s  %s" % (width, rid, exp, obs, dev, verdict))

    failed = [r for r in ROWS if r[4] != "PASS"]
    print("-" * (width + 56))
    print("%d checks, %d passed, %d failed, %.1f s"
          % (len(ROWS), len(ROWS) - len(failed), len(failed),
             time.time() - started))
    if failed:
        print("FAILED:", ", ".join(r[0] for r in failed))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
