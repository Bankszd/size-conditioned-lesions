# Detector-level readout (Table 2, Table 3 H4)

`task_c_v2.py` runs two forward passes over the images.  The first is clean and
caches, for every query, its own predicted box and the head-averaged
cross-attention of every decoder layer.  The second lesions each query inside
its own predicted box and evaluates the resulting detections with the official
COCO evaluator.

Arms: `clean`, `in4`, `out4`, `in_m{0.02,0.05,0.10,0.20,0.40}`, `out_m{0.02,0.40}`,
plus the `prop` and `geo` variants used by Task Y.  `--layers 2` reproduces the
single-layer readout; `--layers all` lesions all six decoder layers, each with
the dose that layer's own clean attention implies.

    python experiments/detector_ap/task_c_v2.py --model detr --cache $DATA_ROOT/coco \
        --config configs/experiments/detector_ap.yaml --out out/ap_detr.jsonl.gz

The size gap is `S = dAP_small - dAP_large` in COCO percentage points, with a
2,000-draw paired whole-image bootstrap.

The per-image detection files are several hundred megabytes and are not part of
this release; `results/tableH4_voc.json` holds the aggregate they produced.
