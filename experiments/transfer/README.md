# Cross-dataset transfer (Table X)

`task_x.py` fits the dose-effect curves on the COCO holdout equal-mass rows
only, freezes them, and applies them to the VOC 2007 test fixed four-cell
anchors.  Three dose measures are compared: `r` (attention-relative), `g`
(cell-count relative) and `m` (absolute mass).

Interval A holds the COCO curve fixed and resamples VOC images; interval B
resamples both datasets and refits the curve inside every draw.

    python experiments/transfer/task_x.py --model detr \
        --coco records/transfer/task_m_detr.jsonl.gz \
        --voc records/relative_dose/r_voc_detr.jsonl.gz --out out/tableX_detr.json

This is a post-hoc transfer test of a frozen curve, not a confirmatory test:
the VOC four-cell gap was already known when it was designed.
