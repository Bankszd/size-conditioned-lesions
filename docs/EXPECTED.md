# What `make verify` checks, and where each number comes from

`make verify` prints one row per published quantity: the value in the paper,
the value recomputed here, the deviation and PASS/FAIL.  It exits non-zero if
any row fails.

Two kinds of block appear in the table.

* **recomputed** - the per-anchor records or bootstrap arrays ship with the
  release, so the frozen analysis script is re-run from scratch.
* **result file** - the block needs per-image detector outputs, which are far
  above the release size budget.  The aggregate JSON that those outputs
  produced ships in `results/`, and `make verify` reads it.  The script that
  produced it is in `experiments/`, and `docs/REPRODUCE.md` gives the command
  to regenerate the inputs on a GPU.

## Recomputed from records

| block | quantity | source |
|---|---|---|
| Table 1 | DETR-R50 and Conditional DETR-R50, `Delta4` and `Delta.02`, with intervals | `records/local_quality/stage7_local_*.jsonl.gz` |
| Table 3 H1, H2 | the same four rows on VOC 2007 test, plus the two decisions | `records/local_quality/task_confirm_*_full.jsonl.gz` |
| Table 3 model comparison | `T` on COCO, VOC 2007 test and VOC 2012 val, with intervals | `records/model_comparison/task_y_boot_*.npz` |
| Task R | R1-R3, including the explained-share difference | `records/relative_dose/r_voc_*.jsonl.gz` |
| Task N (DEIMv2-S) | N1, N2, N3 for DEIMv2-S | `records/deformable/task_d2_n2_deim_voc_officialcfg.jsonl.gz` |
| Table X | measured gap, `r` prediction and `dE_rg` for both dense models | `records/transfer/task_m_*.jsonl.gz`, `records/relative_dose/r_voc_*.jsonl.gz` |

## Read from a result file

| block | file | why |
|---|---|---|
| H3 (both models), N (RT-DETRv2) | `results/tableH3_voc.json`, `results/tableN_voc.json` | RT-DETRv2 per-anchor rows are not in the release |
| H4 | `results/tableH4_voc.json` | per-image detections and 2,000 bootstrap caches |
| D1-D3 | `results/tableD_voc.json` | Deformable DETR VOC rows are not in the release |
| G1, G2 | `results/tableG_voc.json` | per-image deletion curves |
| G5 | `results/tableG5_voc.json` | per-image deletion curves |

## Tolerances

| block | point estimate | interval endpoints | reason |
|---|---|---|---|
| Table 1 | `5e-4` | `2.5e-3` | records regenerated on a different software stack |
| Table 3, Task R, Task N, Table X | printed precision (`1e-5` to `2e-3`) | not compared row by row | bootstrap draws of the shipped arrays |
| Table 3 model comparison (`T`) | `1e-2` AP points | `1e-2` AP points | published to three decimals |
| result-file blocks | `1e-3` to `1e-6` | - | value copied from the frozen aggregate |

## Anonymity and language checks

Two scans were run over the release before packaging.

* **CJK.**  Every text file, and every `.gz` after decompression, was searched
  for `[\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF\u3000-\u303F]`.  Expected and
  actual result: zero matches.
* **Identity.**  File names, file contents, decompressed `.gz` contents, `.npy`
  and `.npz` metadata and the git configuration were searched for user names,
  host names, machine nicknames, cloud-provider domains, private IP addresses,
  email addresses, institution names, remote-login prefixes, port numbers and
  absolute home-directory paths.  Expected and actual result: zero matches.

The word lists and the exact scan commands are printed by the packaging step
and summarised in the release note that accompanies the archive.
