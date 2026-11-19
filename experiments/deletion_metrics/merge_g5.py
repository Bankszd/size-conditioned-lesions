"""Merge G5 shards into one gzip per model and check image coverage."""
import gzip, json, os

OUT = "records/deletion_metrics"
SH = {"detr": ["task_g5_detr_s0", "task_g5_detr_s1"],
      "rt": ["task_g5_rt_s0", "task_g5_rt_s4", "task_g5_rt_s5", "task_g5_rt_s6", "task_g5_rt_s7"]}
res = {}
for model, tags in SH.items():
    merged = os.path.join(OUT, "task_g5_voc_%s_merged.jsonl.gz" % model)
    imgs, rows, full = set(), 0, 0
    with open(merged, "wb") as w:
        for t in tags:
            p = os.path.join(OUT, t + ".jsonl.gz")
            with open(p, "rb") as f:
                w.write(f.read())
    with gzip.open(merged, "rt") as f:
        for line in f:
            r = json.loads(line)
            rows += 1
            if "saliency" in r:
                imgs.add(r["image_id"])
            else:
                full += 1
    res[model] = {"file": merged, "rows": rows, "saliency_images": len(imgs),
                  "extra_rows": full, "first": min(imgs), "last": max(imgs),
                  "size_mb": round(os.path.getsize(merged) / 1e6, 2)}
    print(model, res[model])

for model, tags in SH.items():
    n_img = 0
    n_rows = 0
    for t in tags:
        m = json.load(open(os.path.join(OUT, t + "_meta.json")))
        n_img += m["n_images"]; n_rows += m["n_rows"]
    res[model]["shard_images"] = n_img
    res[model]["shard_rows"] = n_rows
    assert n_img == 4952, (model, n_img)
    assert n_rows == res[model]["rows"] - res[model]["extra_rows"], (model, n_rows, res[model]["rows"], res[model]["extra_rows"])
    assert res[model]["saliency_images"] <= 4952
print("COVERAGE_OK shards cover 4952 images per model; images without any target:",
      {m: 4952 - res[m]["saliency_images"] for m in res})
json.dump(res, open(os.path.join(OUT, "g5_merge_report.json"), "w"), indent=1)


