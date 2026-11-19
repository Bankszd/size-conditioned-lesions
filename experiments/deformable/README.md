# Deformable detectors

Unit of lesion: a `(head, sampling point)` pair.  The mass of a pair is the
head-averaged attention weight, `a / H`, and the per-query masses sum to 1 over
all pairs of all feature levels - the paper reports that check.

Scripts:

* `task_n2_deim_official.py` - DEIMv2-S with the official decoder
  configuration (`interaction_indexes = [3, 7, 11]`).  It reads the config path
  from `$D2_CONFIG` so that the exact file used is recorded in the log.
* `task_d2_rt.py` - RT-DETRv2-R50vd, same protocol.
* `task_d2.py` - the dense-to-deformable translation of the D2 protocol.
* `task_d3.py` - Deformable DETR-R50, including the fixed 2x2 / 3x3 area arms.

Arms: `fixed_inside`, `fixed_outside`, `mass_{0.02..0.40}`, `region2_inside`,
`region2_outside`, `region3_inside`, `region3_outside`.

    D2_CONFIG=configs/deimv2_s_official.json \
    python experiments/deformable/task_n2_deim_official.py --data voc --out out/n_deim.jsonl.gz

The statistics are computed by `experiments/fixed_area/n_voc_analyze.py`.
