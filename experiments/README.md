# Frozen analysis scripts

Every number in the paper comes from a script in this directory.  The scripts
are the versions that produced the registered results; only absolute paths were
rewritten for the release (see `registrations/HASHES.md`).

| folder | paper section | entry point | inputs |
|---|---|---|---|
| `local_quality/` | Table 1, Table 3 H1/H2 | `recompute_tables.py` | `records/local_quality/*.jsonl.gz` |
| `detector_ap/` | Table 3 H4, Table 2, Task Y arms | `task_c_v2.py` | COCO holdout or VOC 2007 test + a checkpoint |
| `model_comparison/` | Table 3 model comparison (D, I, T) | `recompute_y.py` (offline), `task_y_paired.py` (GPU) | `records/model_comparison/*.npz` |
| `deformable/` | deformable-detector lesions (H3, D, N) | `task_n2_deim_official.py`, `task_d2_rt.py`, `task_d3.py` | VOC 2007 test or COCO dev + a checkpoint |
| `fixed_area/` | Task N statistics | `n_voc_analyze.py` | `records/deformable/*.jsonl.gz` |
| `relative_dose/` | Task R | `r_voc_analyze.py` (offline), `task_r.py` (GPU) | `records/relative_dose/*.jsonl.gz` |
| `transfer/` | Table X | `task_x.py` | `records/transfer/*.jsonl.gz` + `records/relative_dose/*.jsonl.gz` |
| `deletion_metrics/` | deletion-curve analyses (G, G5) | `task_g5.py`, `task_g5_rt.py` | VOC 2007 test + a checkpoint |

The offline entry points (`recompute_tables.py`, `recompute_y.py`,
`r_voc_analyze.py`, `n_voc_analyze.py`, `task_x.py`) are the ones `make tables`
runs; they need no GPU, no checkpoint and no image files.

The GPU entry points need the data and checkpoints described in
`docs/REPRODUCE.md`.  They write per-anchor JSONL that the offline scripts then
consume.  Running them reproduces the records bit-for-bit on the same hardware;
see the "Known differences" section of `README.md` for what changes across
software stacks.

Common conventions (paper Section 3):

* decoder cross-attention, zero-based layer index 2;
* token-selection seed 20270930, bootstrap seed 20271001;
* size bins on box area `w * h` with COCO thresholds 32^2 and 96^2;
* `Q = p(class) * IoU(predicted box, ground-truth box)`.
