"""Task R VOC analysis, exactly as CONFIRM_REG_v1_addendum_R.md specifies.

L = Q_clean - Q_lesion ; r = removed mass / clean in-box mass.
Relative curve: 10 equal-width r bins over all equal-mass rows pooled over sizes, a node at each
bin's mean r, plus (0,0); piecewise linear; r above the last node takes the last node's value.
Absolute curve: mean L at each m, plus (0,0); piecewise linear; linear extrapolation beyond 0.80.
Explained share = predicted fixed-4 size gap / measured fixed-4 size gap.
Bootstrap: 5,000 whole-image percentile resamples, PCG64 20271001; both curves are refitted
inside every resample and the share difference is computed within each resample.
"""

# --- release path handling -------------------------------------------------
# The frozen scripts were written inside the authors' working directory.  For
# the release every absolute location is resolved from the environment; the
# defaults reproduce the layout described in docs/REPRODUCE.md.
import os as _os
DATA_ROOT = _os.environ.get("DATA_ROOT", "data")
OUT_ROOT = _os.environ.get("OUT_ROOT", "out")
CKPT_ROOT = _os.environ.get("CKPT_ROOT", "weights")
WORK_ROOT = _os.environ.get("WORK_ROOT", "work")
REPO_ROOT = _os.environ.get("REPO_ROOT", ".")


def _rel(root, tail):
    return _os.path.join(root, tail) if tail else root

# --- end release path handling ---------------------------------------------
import gzip
import json
import sys

import numpy as np

BINS = ["small", "medium", "large"]
MAS = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
FRACS = [0.10, 0.25, 0.50, 0.75]
NB, SEED, NBOOT = 10, 20271001, 5000
MARGIN = 0.005
ARM_NAMES = (["fixed_inside"] + ["mass_%g" % m for m in MAS]
             + ["prop_%g" % f for f in FRACS])


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def pwl(xn, yn, x):
    """Piecewise linear through (xn, yn); linear extrapolation outside the ends."""
    xn = np.asarray(xn, float)
    yn = np.asarray(yn, float)
    x = np.asarray(x, float)
    if xn.size == 1:
        return np.full(x.shape, yn[0], float)
    k = np.clip(np.searchsorted(xn, x) - 1, 0, xn.size - 2)
    x0, x1 = xn[k], xn[k + 1]
    y0, y1 = yn[k], yn[k + 1]
    den = np.where(x1 - x0 > 0, x1 - x0, 1.0)
    return y0 + (y1 - y0) / den * (x - x0)


def pwl_clamp(xn, yn, x):
    """As pwl, but above the last node the value is held at the last node."""
    return pwl(xn, yn, np.minimum(np.asarray(x, float), np.asarray(xn, float)[-1]))


def mean_by_size(rows, arm, key):
    out = []
    for s in range(3):
        v = [r[key] for r in rows if r["arm"] == arm and r["_s"] == s]
        out.append(None if not v else float(np.mean(v)))
    return out


def analyse(model, path):
    rows = load(path)
    imgs = sorted({r["image_id"] for r in rows})
    ix = {v: i for i, v in enumerate(imgs)}
    n = len(imgs)
    for r in rows:
        r["_r"] = r["removed_attention_mass"] / r["clean_in_box_attention_mass"]
        r["_L"] = r["Q_clean"]["quality"] - r["Q_inside"]["quality"]
        r["_i"] = ix[r["image_id"]]
        r["_s"] = BINS.index(r["area_bin"])

    eq = [r for r in rows if r["arm"].startswith("mass_")]
    fix = [r for r in rows if r["arm"] == "fixed_inside"]

    SL = {a: np.zeros((n, 3)) for a in ARM_NAMES}
    CN = {a: np.zeros((n, 3)) for a in ARM_NAMES}
    for r in rows:
        SL[r["arm"]][r["_i"], r["_s"]] += r["_L"]
        CN[r["arm"]][r["_i"], r["_s"]] += 1.0

    ML = np.zeros((n, len(MAS)))
    MC = np.zeros((n, len(MAS)))
    BL = np.zeros((n, NB))
    BR = np.zeros((n, NB))
    BC = np.zeros((n, NB))
    for r in eq:
        mi = int(np.argmin([abs(m - r["nominal_mass"]) for m in MAS]))
        ML[r["_i"], mi] += r["_L"]
        MC[r["_i"], mi] += 1.0
        k = min(NB - 1, int(np.floor(r["_r"] * NB)))
        BL[r["_i"], k] += r["_L"]
        BR[r["_i"], k] += r["_r"]
        BC[r["_i"], k] += 1.0

    fix_i = np.array([r["_i"] for r in fix])
    fix_s = np.array([r["_s"] for r in fix])
    fix_m = np.array([r["removed_attention_mass"] for r in fix])
    fix_r = np.array([r["_r"] for r in fix])

    def gap_of(arm, w):
        sl = w @ SL[arm]
        cn = w @ CN[arm]
        if cn[0] <= 0 or cn[2] <= 0:
            return np.nan
        return sl[0] / cn[0] - sl[2] / cn[2]

    def curves(w):
        ml = w @ ML
        mc = w @ MC
        ya = np.concatenate(([0.0], np.where(mc > 0, ml / np.maximum(mc, 1e-12), np.nan)))
        xa = np.array([0.0] + MAS)
        ok = np.isfinite(ya)
        bl = w @ BL
        br = w @ BR
        bc = w @ BC
        xr = np.where(bc > 0, br / np.maximum(bc, 1e-12), np.nan)
        yr = np.where(bc > 0, bl / np.maximum(bc, 1e-12), np.nan)
        okr = np.isfinite(xr) & np.isfinite(yr) & (bc > 0)
        xr, yr = xr[okr], yr[okr]
        o = np.argsort(xr)
        xr = np.concatenate(([0.0], xr[o]))
        yr = np.concatenate(([0.0], yr[o]))
        keep = np.concatenate(([True], np.diff(xr) > 0))
        return xa[ok], ya[ok], xr[keep], yr[keep]

    def pred_gap(w, xn, yn, x, clamp):
        p = pwl_clamp(xn, yn, x) if clamp else pwl(xn, yn, x)
        ws = w[fix_i]
        a = (ws * (fix_s == 0) * p).sum() / max((ws * (fix_s == 0)).sum(), 1e-12)
        b = (ws * (fix_s == 2) * p).sum() / max((ws * (fix_s == 2)).sum(), 1e-12)
        return a - b

    def shares(w):
        xa, ya, xr, yr = curves(w)
        obs = gap_of("fixed_inside", w)
        if not np.isfinite(obs) or obs == 0:
            return np.nan, np.nan, obs
        return (pred_gap(w, xr, yr, fix_r, True) / obs,
                pred_gap(w, xa, ya, fix_m, False) / obs, obs)

    ones = np.ones(n)
    sr, sa, obs_pt = shares(ones)
    d_pt = {a: gap_of(a, ones) for a in ARM_NAMES}
    xa, ya, xr, yr = curves(ones)

    rng = np.random.Generator(np.random.PCG64(SEED))
    W = rng.integers(0, n, size=(NBOOT, n))
    d_boot = {a: np.full(NBOOT, np.nan) for a in ARM_NAMES}
    obs_b = np.full(NBOOT, np.nan)
    sr_b = np.full(NBOOT, np.nan)
    sa_b = np.full(NBOOT, np.nan)
    ov_b = np.zeros(NBOOT)
    for b in range(NBOOT):
        w = np.bincount(W[b], minlength=n).astype(float)
        for a in ARM_NAMES:
            d_boot[a][b] = gap_of(a, w)
        sr_b[b], sa_b[b], obs_b[b] = shares(w)
        _, _, xrb, _ = curves(w)
        ov_b[b] = float((fix_r > xrb[-1]).sum())
    diff_b = sr_b - sa_b

    out = {"model": model, "path": path, "n_images": n, "n_rows": len(rows),
           "n_equal_mass_rows": len(eq), "n_fixed_rows": len(fix),
           "fixed_n_by_size": {b: int(CN["fixed_inside"][:, s].sum())
                               for s, b in enumerate(BINS)},
           "abs_curve": {"x": [float(v) for v in xa], "y": [float(v) for v in ya]},
           "rel_curve": {"x": [float(v) for v in xr], "y": [float(v) for v in yr]},
           "n_fixed_above_last_rel_node": int((fix_r > xr[-1]).sum()),
           "n_fixed_above_last_rel_node_boot_mean": float(ov_b.mean()),
           "share_relative": float(sr), "share_relative_ci": ci(sr_b),
           "share_absolute": float(sa), "share_absolute_ci": ci(sa_b),
           "share_diff": float(sr - sa), "share_diff_ci": ci(diff_b),
           "n_boot_valid_diff": int(np.isfinite(diff_b).sum()),
           "measured_fixed_gap": float(obs_pt), "measured_fixed_gap_ci": ci(obs_b),
           "arms": {}}
    for a in ARM_NAMES:
        sl = SL[a].sum(0)
        cn = CN[a].sum(0)
        out["arms"][a] = {
            "n_by_size": [int(v) for v in cn],
            "mean_L_by_size": [None if cn[s] == 0 else float(sl[s] / cn[s]) for s in range(3)],
            "delta": None if not np.isfinite(d_pt[a]) else float(d_pt[a]),
            "delta_ci": ci(d_boot[a]),
            "mean_removed_by_size": mean_by_size(rows, a, "removed_attention_mass"),
            "mean_rel_by_size": mean_by_size(rows, a, "_r"),
            "mean_clean_mass_by_size": mean_by_size(rows, a, "clean_in_box_attention_mass")}
    return out


def _verdict(vals, kind):
    """vals: list of intervals, one per model. kind: 'gt0', 'lt0', 'inside'."""
    if any(v is None for v in vals):
        return "inconclusive"
    if kind == "gt0":
        if all(v[0] > 0 for v in vals):
            return "confirmed"
        if any(v[1] < 0 for v in vals):
            return "contradicted"
        return "inconclusive"
    if kind == "lt0":
        if all(v[1] < 0 for v in vals):
            return "confirmed"
        if any(v[0] > 0 for v in vals):
            return "contradicted"
        return "inconclusive"
    if all(v[0] >= -MARGIN and v[1] <= MARGIN for v in vals):
        return "confirmed"
    if any(v[1] < -MARGIN or v[0] > MARGIN for v in vals):
        return "contradicted"
    return "inconclusive"


if __name__ == "__main__":
    paths = {"detr": sys.argv[1], "conditional_detr": sys.argv[2]}
    res = {}
    for k, p in paths.items():
        res[k] = analyse(k, p)
        print("done", k, flush=True)
    d, c = res["detr"], res["conditional_detr"]
    res["verdicts"] = {
        "R1": _verdict([d["share_diff_ci"], c["share_diff_ci"]], "gt0"),
        "R2": _verdict([d["arms"]["prop_0.1"]["delta_ci"],
                        c["arms"]["prop_0.1"]["delta_ci"]], "inside"),
        "R3": _verdict([d["arms"]["prop_0.75"]["delta_ci"],
                        c["arms"]["prop_0.75"]["delta_ci"]], "gt0")}
    out_path = sys.argv[3] if len(sys.argv) > 3 else _rel(OUT_ROOT, "r_voc_results.json")
    with open(out_path, "w", encoding="utf8") as f:
        f.write(json.dumps(res, indent=1, default=str))
    for k in ("detr", "conditional_detr"):
        v = res[k]
        print(k, "images", v["n_images"], "rows", v["n_rows"], "fixed n", v["fixed_n_by_size"])
        print("  observed fixed gap", round(v["measured_fixed_gap"], 6),
              [round(x, 6) for x in v["measured_fixed_gap_ci"]])
        print("  share rel", round(v["share_relative"], 4),
              [round(x, 4) for x in v["share_relative_ci"]],
              "abs", round(v["share_absolute"], 4),
              [round(x, 4) for x in v["share_absolute_ci"]],
              "diff", round(v["share_diff"], 4),
              [round(x, 4) for x in v["share_diff_ci"]])
        print("  above last rel node", v["n_fixed_above_last_rel_node"])
        for a in ARM_NAMES[1:]:
            dd = v["arms"][a]
            print("  ", a, dd["n_by_size"],
                  "delta", None if dd["delta"] is None else round(dd["delta"], 6),
                  "ci", None if dd["delta_ci"] is None
                  else [round(x, 6) for x in dd["delta_ci"]])
    print("verdicts", res["verdicts"])
