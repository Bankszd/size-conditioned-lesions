"""In-box dose fraction and patch fraction at the reporting nodes, per model."""
import gzip, json
import numpy as np

FILES = {"detr": "records/deletion_metrics/task_g5_voc_detr_merged.jsonl.gz",
         "rt": "records/deletion_metrics/task_g5_voc_rt_merged.jsonl.gz"}
NODES = (0, 1, 5, 10, 20, 50)
out = {}
for model, p in FILES.items():
    rows = []
    with gzip.open(p, "rt") as f:
        for l in f:
            r = json.loads(l)
            if "saliency" in r:
                rows.append(r)
    d = {}
    for sal in ("random", "attn", "grad"):
        for b in ("small", "medium", "large"):
            rs = [r for r in rows if r["saliency"] == sal and r["bin"] == b]
            if not rs:
                continue
            for nd in NODES:
                d["%s|%s|%d" % (sal, b, nd)] = {
                    "dose": float(np.mean([r["box_dose_frac"][nd] for r in rs])),
                    "patch": float(np.mean([r["box_removed_frac"][nd] for r in rs]))}
    out[model] = d
json.dump(out, open("records/deletion_metrics/g5_dose_patch.json", "w"), indent=1)
for model in out:
    print("==", model)
    for sal in ("random",):
        for b in ("small", "medium", "large"):
            print(" %-6s %-6s" % (sal, b), " ".join(
                "%d:%.4f/%.4f" % (nd, out[model]["%s|%s|%d" % (sal, b, nd)]["dose"],
                                  out[model]["%s|%s|%d" % (sal, b, nd)]["patch"]) for nd in NODES))
