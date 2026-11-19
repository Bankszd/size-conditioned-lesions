"""Task X: transfer of frozen COCO dose-effect curves to VOC 2007 test.

Dose measures per row: r = removed / clean in-box mass; g = deleted in-box cells / box cells
(partial cell counted by its removed fraction); m = absolute removed mass.
Curves are fitted on COCO holdout equal-mass rows only, then frozen and applied to the VOC
fixed four-cell arm. No VOC row enters any fit.
"""
import argparse, gzip, json
import numpy as np

MASSES = (0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80)
NBIN = 10

def load(path, arms):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] not in arms:
                continue
            clean = float(r["clean_in_box_attention_mass"])
            rem = float(r["removed_attention_mass"])
            bc = float(r["box_cells"])
            if clean <= 0 or bc <= 0:
                continue
            cells = float(r["attenuated_cell_count"]) + float(r["fractional_cell_count"])
            rows.append((int(r["image_id"]), r["area_bin"], float(r["Q_clean"]["quality"]),
                         float(r["Q_inside"]["quality"]), clean, rem, cells, bc))
    return rows

def pack(rows):
    img = np.array([x[0] for x in rows])
    b = np.array([x[1] for x in rows])
    L = np.array([x[2] - x[3] for x in rows])
    clean = np.array([x[4] for x in rows])
    rem = np.array([x[5] for x in rows])
    cells = np.array([x[6] for x in rows])
    bc = np.array([x[7] for x in rows])
    r = np.clip(rem / clean, 0.0, 1.0)
    g = np.clip(cells / bc, 0.0, 1.0)
    return {"img": img, "bin": b, "L": L, "r": r, "g": g, "m": rem}

def wmean(num, den):
    return num / np.maximum(den, 1e-12)

def fit_binned(d, key, weights=None):
    x = d[key]
    L = d["L"]
    w = np.ones_like(L) if weights is None else weights
    edges = np.linspace(0, 1, NBIN + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, NBIN - 1)
    xs, ys = [0.0], [0.0]
    for k in range(NBIN):
        s = idx == k
        sw = w[s].sum()
        if sw <= 1e-9:
            continue
        xs.append(float((x[s] * w[s]).sum() / sw))
        ys.append(float((L[s] * w[s]).sum() / sw))
    xs, ys = np.array(xs), np.array(ys)
    o = np.argsort(xs)
    return xs[o], ys[o]

def fit_mass(d, weights=None):
    w = np.ones_like(d["L"]) if weights is None else weights
    xs, ys = [0.0], [0.0]
    for mm in MASSES:
        s = np.abs(d["m"] - mm) < 1e-6
        sw = w[s].sum()
        if sw <= 1e-9:
            continue
        xs.append(mm); ys.append(float((d["L"][s] * w[s]).sum() / sw))
    return np.array(xs), np.array(ys)

def predict(x, xs, ys, clamp=True):
    if clamp:
        return np.interp(x, xs, ys, left=ys[0], right=ys[-1])
    out = np.interp(x, xs, ys, left=ys[0], right=ys[-1])
    hi = x > xs[-1]
    if hi.any():
        slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-12)
        out[hi] = ys[-1] + slope * (x[hi] - xs[-1])
    return out

def gaps(L, bins, w):
    sm = bins == "small"; lg = bins == "large"
    ws = w[sm].sum(); wl = w[lg].sum()
    if ws <= 0 or wl <= 0:
        return np.nan
    return float((L[sm] * w[sm]).sum() / ws - (L[lg] * w[lg]).sum() / wl)

def image_weights(draw, n_img):
    return np.bincount(draw, minlength=n_img).astype(float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--voc", required=True)
    ap.add_argument("--n-a", type=int, default=5000)
    ap.add_argument("--n-b", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    coco_arms = tuple("mass_%.2f" % m for m in MASSES) + tuple("mass_%.1f" % m for m in MASSES)
    coco = pack(load(args.coco, set(coco_arms)))
    voc = pack(load(args.voc, {"fixed_inside"}))
    print("rows coco/voc", len(coco["L"]), len(voc["L"]), flush=True)

    xs_r, ys_r = fit_binned(coco, "r")
    xs_g, ys_g = fit_binned(coco, "g")
    xs_m, ys_m = fit_mass(coco)

    pred_r = predict(voc["r"], xs_r, ys_r)
    pred_g = predict(voc["g"], xs_g, ys_g)
    pred_m = predict(voc["m"], xs_m, ys_m, clamp=False)

    n_img_v = int(voc["img"].max()) + 1
    wv = np.ones_like(voc["L"])
    obs = gaps(voc["L"], voc["bin"], wv)
    pr = gaps(pred_r, voc["bin"], wv)
    pg = gaps(pred_g, voc["bin"], wv)
    pm = gaps(pred_m, voc["bin"], wv)
    beyond_r = int((voc["r"] > xs_r[-1]).sum()); beyond_g = int((voc["g"] > xs_g[-1]).sum())
    beyond_m = int((voc["m"] > xs_m[-1]).sum())
    print("OBS %+.6f  pred_r %+.6f pred_g %+.6f pred_m %+.6f" % (obs, pr, pg, pm), flush=True)
    print("BEYOND r %d g %d m %d" % (beyond_r, beyond_g, beyond_m), flush=True)

    vimg = voc["img"]
    rng = np.random.Generator(np.random.PCG64(20271001))
    dE_rg = np.zeros(args.n_a); dE_rm = np.zeros(args.n_a)
    obs_a = np.zeros(args.n_a); pr_a = np.zeros(args.n_a); pg_a = np.zeros(args.n_a); pm_a = np.zeros(args.n_a)
    for i in range(args.n_a):
        w = image_weights(rng.integers(0, n_img_v, n_img_v), n_img_v)[vimg]
        o = gaps(voc["L"], voc["bin"], w)
        prr = gaps(pred_r, voc["bin"], w); pgg = gaps(pred_g, voc["bin"], w); pmm = gaps(pred_m, voc["bin"], w)
        obs_a[i], pr_a[i], pg_a[i], pm_a[i] = o, prr, pgg, pmm
        dE_rg[i] = abs(pgg - o) - abs(prr - o)
        dE_rm[i] = abs(pmm - o) - abs(prr - o)
    print("BOOT_A_DONE", flush=True)

    n_img_c = int(coco["img"].max()) + 1
    cimg = coco["img"]
    dE_rg_b = np.zeros(args.n_b); dE_rm_b = np.zeros(args.n_b)
    for i in range(args.n_b):
        wc = image_weights(rng.integers(0, n_img_c, n_img_c), n_img_c)[cimg]
        xr, yr = fit_binned(coco, "r", wc); xg, yg = fit_binned(coco, "g", wc); xm, ym = fit_mass(coco, wc)
        wv2 = image_weights(rng.integers(0, n_img_v, n_img_v), n_img_v)[vimg]
        o = gaps(voc["L"], voc["bin"], wv2)
        prr = gaps(predict(voc["r"], xr, yr), voc["bin"], wv2)
        pgg = gaps(predict(voc["g"], xg, yg), voc["bin"], wv2)
        pmm = gaps(predict(voc["m"], xm, ym, clamp=False), voc["bin"], wv2)
        dE_rg_b[i] = abs(pgg - o) - abs(prr - o)
        dE_rm_b[i] = abs(pmm - o) - abs(prr - o)
    print("BOOT_B_DONE", flush=True)

    def ci(v, q=(2.5, 97.5)):
        return [float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))]

    out = {"model": args.model, "n_rows": {"coco": len(coco["L"]), "voc": len(voc["L"])},
           "curves": {"r": {"x": xs_r.tolist(), "y": ys_r.tolist()},
                      "g": {"x": xs_g.tolist(), "y": ys_g.tolist()},
                      "m": {"x": xs_m.tolist(), "y": ys_m.tolist()}},
           "beyond_last_node": {"r": beyond_r, "g": beyond_g, "m": beyond_m},
           "observed": obs,
           "observed_ci_A": ci(obs_a),
           "predicted": {"r": pr, "g": pg, "m": pm},
           "predicted_ci_A": {"r": ci(pr_a), "g": ci(pg_a), "m": ci(pm_a)},
           "error": {"r": pr - obs, "g": pg - obs, "m": pm - obs},
           "dE_rg": {"observed": abs(pg - obs) - abs(pr - obs), "ci_A": ci(dE_rg), "ci_B": ci(dE_rg_b)},
           "dE_rm": {"observed": abs(pm - obs) - abs(pr - obs), "ci_A": ci(dE_rm), "ci_B": ci(dE_rm_b)},
           "n_draws": {"A": args.n_a, "B": args.n_b}}
    json.dump(out, open(args.out, "w"), indent=1, sort_keys=True)
    print("WROTE", args.out, flush=True)
    print("dE_rg %+.6f A[%+.6f,%+.6f] B[%+.6f,%+.6f]" % (out["dE_rg"]["observed"], *out["dE_rg"]["ci_A"], *out["dE_rg"]["ci_B"]))
    print("dE_rm %+.6f A[%+.6f,%+.6f] B[%+.6f,%+.6f]" % (out["dE_rm"]["observed"], *out["dE_rm"]["ci_A"], *out["dE_rm"]["ci_B"]))

if __name__ == "__main__":
    main()
