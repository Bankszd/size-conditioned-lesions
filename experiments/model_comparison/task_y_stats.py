"""Task Y statistics: S per combo, D, I, T and T-prime, all from shared draws."""

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
import argparse, json, os
import numpy as np

def load(path):
    zs, obs = {}, {}
    for one in path.split(","):
        z = np.load(one, allow_pickle=True)
        if "arr_0" in z.files:
            z = z["arr_0"].item()
        zs.update({k: v for k, v in z.items()})
        m = json.load(open(one.replace(".npz", "_meta.json")))
        obs.update(m["observed"])
    return zs, {"observed": obs}

def arr(z, ver, arm, metric):
    return z["%s|%s|%d" % (ver, arm, metric)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detr", default="_rel(OUT_ROOT, "")/task_y_boot_y_detr.npz")
    ap.add_argument("--cond", default="_rel(OUT_ROOT, "")/task_y_boot_y_cond.npz")
    ap.add_argument("--arms", default="four,in4,in_m0.02,in_prop0.50")
    ap.add_argument("--out", default="_rel(OUT_ROOT, "")/task_y_stats.json")
    ap.add_argument("--md", default="_rel(OUT_ROOT, "")/tables_y.md")
    args = ap.parse_args()
    alias = {"four": "in4", "m02": "in_m0.02", "prop50": "in_prop0.50"}
    zd, md_ = load(args.detr)
    zc, mc = load(args.cond)
    versions = ("layer2", "all_layers")
    rules = [r for r in args.arms.split(",") if r]
    models = {"detr": (zd, md_), "conditional_detr": (zc, mc)}

    res = {"point": {}, "S": {}, "D": {}, "I": {}, "T": {}}
    Sd = {}
    for model, (z, meta) in models.items():
        for ver in versions:
            if "%s|clean|0" % ver not in z:
                continue
            clean_aps = arr(z, ver, "clean", 1); clean_apl = arr(z, ver, "clean", 2)
            for rule in rules:
                arm = alias[rule]
                key = "%s|%s|%d" % (ver, arm, 0)
                if key not in z:
                    continue
                s = (arr(z, ver, arm, 1) - clean_aps) - (arr(z, ver, arm, 2) - clean_apl)
                Sd[(model, ver, rule)] = s
                o = meta["observed"]
                c_ap, c_aps, c_apl = (o["%s|clean|%d" % (ver, mm)] for mm in (0, 1, 2))
                a_ap, a_aps, a_apl = (o["%s|%s|%d" % (ver, arm, mm)] for mm in (0, 1, 2))
                p = {"clean": [c_ap, c_aps, c_apl], "arm": [a_ap, a_aps, a_apl],
                     "S_observed": (a_aps - c_aps) - (a_apl - c_apl),
                     "S_mean": float(s.mean()),
                     "S_ci": [float(np.quantile(s, .025)), float(np.quantile(s, .975))]}
                res["point"]["%s|%s|%s" % (model, ver, rule)] = p
                res["S"]["%s|%s|%s" % (model, ver, rule)] = {
                    "observed": p["S_observed"], "mean": p["S_mean"], "ci95": p["S_ci"]}

    for rule in rules:
        for ver in versions:
            kd = ("detr", ver, rule); kc = ("conditional_detr", ver, rule)
            if kd in Sd and kc in Sd:
                d = Sd[kd] - Sd[kc]
                dobs = res["S"]["%s|%s|%s" % kd]["observed"] - res["S"]["%s|%s|%s" % kc]["observed"]
                res["D"]["%s|%s" % (ver, rule)] = {
                    "observed": float(dobs), "mean": float(d.mean()),
                    "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
    for rule in rules:
        kl = "layer2|" + rule; ka = "all_layers|" + rule
        if kl in res["D"] and ka in res["D"]:
            il = res["D"][kl]; ia = res["D"][ka]
            iobs = ia["observed"] - il["observed"]
            dd = (Sd[("detr", "all_layers", rule)] - Sd[("conditional_detr", "all_layers", rule)]) - \
                 (Sd[("detr", "layer2", rule)] - Sd[("conditional_detr", "layer2", rule)])
            res["I"][rule] = {"observed": float(iobs), "mean": float(dd.mean()),
                              "ci95": [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))]}
    for name, a, b in (("T", "four", "prop50"), ("T_prime", "four", "m02")):
        if a in res["I"] and b in res["I"]:
            ia, ib = res["I"][a], res["I"][b]
            dd = (Sd[("detr", "all_layers", a)] - Sd[("conditional_detr", "all_layers", a)] -
                  Sd[("detr", "layer2", a)] + Sd[("conditional_detr", "layer2", a)]) - \
                 (Sd[("detr", "all_layers", b)] - Sd[("conditional_detr", "all_layers", b)] -
                  Sd[("detr", "layer2", b)] + Sd[("conditional_detr", "layer2", b)])
            res["T"][name] = {"observed": ia["observed"] - ib["observed"], "mean": float(dd.mean()),
                              "ci95": [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))]}
    json.dump(res, open(args.out, "w"), indent=1, sort_keys=True)
    print("WROTE", args.out)

    L = []
    L.append("## Table Y1 - size-gap contrasts (COCO holdout, 4,000 images; 2,000 paired draws)\n")
    L.append("Units: AP percentage points (S is a difference of AP_S and AP_L).")
    L.append("| rule | D_L3 | D_ALL | I |")
    L.append("|---|---|---|---|")
    for rule in rules:
        def f(k):
            v = res["D"].get(k)
            return "n/a" if not v else "%+.3f [%+.3f, %+.3f]" % (v["observed"] * 100, v["ci95"][0] * 100, v["ci95"][1] * 100)
        i = res["I"].get(rule)
        istr = "n/a" if not i else "%+.3f [%+.3f, %+.3f]" % (i["observed"] * 100, i["ci95"][0] * 100, i["ci95"][1] * 100)
        L.append("| %s | %s | %s | %s |" % (rule, f("layer2|" + rule), f("all_layers|" + rule), istr))
    for name in ("T", "T_prime"):
        if name in res["T"]:
            v = res["T"][name]
            L.append("| %s | - | - | %+.3f [%+.3f, %+.3f] |" % (name, v["observed"] * 100, v["ci95"][0] * 100, v["ci95"][1] * 100))
    L.append("")
    L.append("## Table Y2 - AP readings\n")
    L.append("| model | depth | rule | AP | AP_S | AP_L | dAP | dAP_S | dAP_L | S [95% CI] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for model in models:
        for ver in versions:
            for rule in rules:
                k = "%s|%s|%s" % (model, ver, rule)
                if k not in res["point"]:
                    continue
                p = res["point"][k]
                L.append("| %s | %s | %s | %.3f | %.3f | %.3f | %+.3f | %+.3f | %+.3f | %+.3f [%+.3f, %+.3f] |" % (
                    model, ver, rule, p["arm"][0] * 100, p["arm"][1] * 100, p["arm"][2] * 100,
                    (p["arm"][0] - p["clean"][0]) * 100, (p["arm"][1] - p["clean"][1]) * 100,
                    (p["arm"][2] - p["clean"][2]) * 100,
                    p["S_observed"] * 100, p["S_ci"][0] * 100, p["S_ci"][1] * 100))
    open(args.md, "w").write("\n".join(L) + "\n")
    print("WROTE", args.md)
    for name in ("T", "T_prime"):
        if name in res["T"]:
            v = res["T"][name]
            print(name, "%+.5f [%+.5f, %+.5f]" % (v["observed"], v["ci95"][0], v["ci95"][1]))

if __name__ == "__main__":
    main()
