# Local quality (Table 1, Table 3 H1/H2)

`recompute_tables.py` reads the per-anchor records and prints

* Table 1 - COCO holdout, DETR-R50 and Conditional DETR-R50:
  `Delta4 = Q_outside - Q_inside` (fixed four cells) and
  `Delta.02 = Q_clean - Q_inside` (equal mass 0.02);
* Table 3 - the same two quantities on VOC 2007 test, which are the inputs to
  hypotheses H1 and H2.

The statistic is always `E[.|small] - E[.|large]` with a whole-image
percentile bootstrap (5,000 resamples, PCG64 seed 20271001).

    cd records/local_quality
    python ../../experiments/local_quality/recompute_tables.py --json out.json

Record columns used: `arm`, `image_id`, `annotation_id`, `area_bin`,
`clean_local.quality`, `perturbed_local.quality`.
