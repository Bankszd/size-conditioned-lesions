"""Recompute Task Y (and the geo50 / VOC-Y variants) from the shipped bootstrap arrays.

Everything here is derived from per-draw AP / AP_S / AP_L arrays written by
task_y_paired.py and task_y_paired_voc.py.  All arms share one whole-image
resampling matrix (PCG64 seed 20271001), so the contrasts are formed draw by draw
without touching predictions or a GPU.

Usage (defaults reproduce Table Y1 of the paper):
    python recompute_y.py
    python recompute_y.py --npz task_y_boot_y_detr.npz task_y_boot_y_detr_prop.npz \
                          --npz task_y_boot_y_cond.npz task_y_boot_y_cond_prop.npz

Each --npz must have a companion _meta.json with the fields produced by
task_y_paired.py (model, versions, observed).  Several files for the same model
are merged; the "clean" key is shared between them.
"""
import argparse, json, os, sys
import numpy as np


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    if "arr_0" in z.files and z["arr_0"].dtype == object:
        return z["arr_0"].item()
    return {k: z[k] for k in z.files}


def model_key(name):
    n = str(name).lower()
    return "conditional_detr" if n.startswith("cond") else "detr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="*", action="extend", default=[],
                    help="bootstrap npz files; defaults to the four Task Y files")
    args = ap.parse_args()
    files = args.npz or ["task_y_boot_y_detr.npz", "task_y_boot_y_detr_prop.npz",
                         "task_y_boot_y_cond.npz", "task_y_boot_y_cond_prop.npz"]

    boot, obs = {}, {}
    for path in files:
        meta_path = path.replace(".npz", "_meta.json")
        if not os.path.exists(meta_path):
            sys.exit("missing meta: " + meta_path)
        data = load_npz(path)
        meta = json.load(open(meta_path))
        mk = model_key(meta["model"])
        for k, v in data.items():
            boot[(mk,) + tuple(k.split("|"))] = np.asarray(v)
        for k, v in meta["observed"].items():
            obs[(mk,) + tuple(k.split("|"))] = float(v)

    arms, versions = set(), set()
    for key in boot:
        versions.add(key[1]); arms.add(key[2])
    versions = [v for v in ("layer2", "all_layers") if v in versions]
    arms = sorted(arms - {"clean"})

    def gap(model, version, arm):
        b = ((boot[(model, version, arm, "1")] - boot[(model, version, "clean", "1")]) -
             (boot[(model, version, arm, "2")] - boot[(model, version, "clean", "2")]))
        o = ((obs[(model, version, arm, "1")] - obs[(model, version, "clean", "1")]) -
             (obs[(model, version, arm, "2")] - obs[(model, version, "clean", "2")]))
        return b, o

    def ci(x, nd=3):
        return "[%+.*f, %+.*f]" % (nd, np.percentile(x, 2.5) * 100, nd, np.percentile(x, 97.5) * 100)

    print("models: %s | arms: %s | draws: %d" % (sorted({k[0] for k in boot}), arms,
                                                 len(boot[(sorted({k[0] for k in boot})[0], versions[0], "clean", "0")])))
    print()
    print("=== Table Y2 (AP / AP_S / AP_L, percentage points) ===")
    print("%-18s %-11s %-9s %8s %8s %8s %8s %8s %8s  %s" %
          ("model", "depth", "rule", "AP", "AP_S", "AP_L", "dAP", "dAP_S", "dAP_L", "S [95% CI]"))
    for model in ("detr", "conditional_detr"):
        if (model, versions[0], "clean", "0") not in boot:
            continue
        for version in versions:
            clean = {m: obs[(model, version, "clean", m)] * 100 for m in ("0", "1", "2")}
            print("%-18s %-11s %-9s %8.3f %8.3f %8.3f %8s %8s %8s  %s" %
                  (model, version, "clean", clean["0"], clean["1"], clean["2"], "-", "-", "-", "-"))
            for arm in arms:
                vals = {m: obs[(model, version, arm, m)] * 100 for m in ("0", "1", "2")}
                b, o = gap(model, version, arm)
                print("%-18s %-11s %-9s %8.3f %8.3f %8.3f %+8.3f %+8.3f %+8.3f  %+.3f %s" %
                      (model, version, arm, vals["0"], vals["1"], vals["2"],
                       vals["0"] - clean["0"], vals["1"] - clean["1"], vals["2"] - clean["2"],
                       o * 100, ci(b)))

    print()
    print("=== Table Y1 (size-gap contrasts) ===")
    print("%-11s %-32s %-32s %s" % ("rule", "D_L3", "D_ALL", "I"))
    D = {}
    for arm in arms:
        cells = []
        for version in versions:
            b1, o1 = gap("detr", version, arm)
            b2, o2 = gap("conditional_detr", version, arm)
            D[(version, arm)] = (b1 - b2, o1 - o2)
            cells.append("%+.3f %s" % ((o1 - o2) * 100, ci(b1 - b2)))
        if len(versions) == 2:
            bi = D[(versions[1], arm)][0] - D[(versions[0], arm)][0]
            oi = D[(versions[1], arm)][1] - D[(versions[0], arm)][1]
            D[("I", arm)] = (bi, oi)
            cells.append("%+.3f %s" % (oi * 100, ci(bi)))
        print("%-11s %-32s %-32s %s" % (arm, cells[0], cells[1], cells[2] if len(cells) > 2 else ""))

    pairs = [("T     = I_four - I_prop50", "in4", "in_prop0.50"),
             ("T'    = I_four - I_m02", "in4", "in_m0.02"),
             ("T_g   = I_geo50 - I_prop50", "in_geo0.50", "in_prop0.50"),
             ("T''   = I_four - I_geo50", "in4", "in_geo0.50"),
             ("T_mg  = I_geo50 - I_m02", "in_geo0.50", "in_m0.02")]
    print()
    for label, a, b in pairs:
        if ("I", a) in D and ("I", b) in D:
            bb = D[("I", a)][0] - D[("I", b)][0]
            oo = D[("I", a)][1] - D[("I", b)][1]
            print("%-28s %+.3f %s pp" % (label, oo * 100, ci(bb)))


if __name__ == "__main__":
    main()
