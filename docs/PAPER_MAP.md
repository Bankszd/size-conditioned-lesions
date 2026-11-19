# Paper to code

Every row names the result file that holds the numbers, the record file the
numbers came from, the script that computes them and the command that reruns
it.  "offline" means `make tables` covers it; "GPU" means the command needs a
checkpoint and the corresponding dataset (`docs/REPRODUCE.md`).

| paper item | result file | records | script | command |
|---|---|---|---|---|
| Table 1 (COCO local quality) | `results/table1_table3_aggregates.json` | `records/local_quality/stage7_local_*.jsonl.gz` | `experiments/local_quality/recompute_tables.py` | offline |
| Table 2 (detector AP, COCO) | `results/geometry_control_stats.json` | per-image detections (not shipped) | `experiments/detector_ap/task_c_v2.py` | GPU |
| Table 3 H1, H2 (VOC local quality) | recomputed | `records/local_quality/task_confirm_*_full.jsonl.gz` | `experiments/local_quality/recompute_tables.py` | offline |
| Table 3 H3 (deformable null) | `results/tableH3_voc.json` | `records/deformable/*` (DEIMv2-S), aggregate (RT-DETRv2) | `experiments/deformable/*` | GPU |
| Table 3 H4 (detector AP, VOC) | `results/tableH4_voc.json` | `records/detector_ap/*_summary.json` | `experiments/detector_ap/task_c_v2.py` | GPU |
| Table 3 model comparison | recomputed | `records/model_comparison/task_y_boot_*.npz` | `experiments/model_comparison/recompute_y.py` | offline |
| Table 4 (Task R) | `results/tableR_stats.json` | `records/relative_dose/r_voc_*.jsonl.gz` | `experiments/relative_dose/r_voc_analyze.py` | offline |
| Table 5 (Task N) | `results/tableN_stats.json`, `results/tableN_voc.json` | `records/deformable/*officialcfg.jsonl.gz` | `experiments/fixed_area/n_voc_analyze.py` | offline |
| Table 6 (Task D) | `results/tableD_voc.json` | Deformable DETR VOC rows (not shipped) | `experiments/deformable/task_d3.py` | GPU |
| Table X (frozen-curve transfer) | `results/task_x_*.json` | `records/transfer/task_m_*.jsonl.gz`, `records/relative_dose/r_voc_*.jsonl.gz` | `experiments/transfer/task_x.py` | offline |
| Deletion-metric tables (G, G2, G5) | `results/tableG_voc.json`, `results/tableG5_voc.json` | `records/deletion_metrics/*.jsonl.gz` | `experiments/deletion_metrics/task_g5*.py` | GPU |
| Dose-gap figure | `results/tableH4_voc.json`, `results/tableN_voc.json` | as above | as above | offline + GPU |

## Where the drawing data lives

The paper's figures are drawn from the same JSON files.  The per-step deletion
curves used in the figures are in `records/deletion_metrics/*.jsonl.gz`, keyed
by `image_id`, `saliency`, `axis`, `step` and `score`.

## Conventions shared by every row

* decoder cross-attention, zero-based layer index 2;
* anchor selection seed 20270930, bootstrap seed 20271001;
* whole-image resampling: every anchor of a drawn image moves with it, and an
  image drawn `k` times enters `k` times;
* `Q = p(class) * IoU(predicted box, ground-truth box)`;
* COCO size bins on `w * h` with thresholds 32^2 and 96^2.
