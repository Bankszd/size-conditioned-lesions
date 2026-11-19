# Four Cells Are Not the Same Dose: size-conditioned spatial lesions in DETR-family detectors

This repository contains the code, the per-anchor records and the bootstrap
arrays behind every table and figure of the paper.  All of the released
analysis runs on CPU in about three minutes; no checkpoint, no image file and
no GPU is required to rebuild the tables.

## Key findings

* **A fixed number of cells is not a fixed dose.** In DETR-R50, four random
  in-box cells remove on average `0.396` of the matched query's total clean
  attention for eligible small objects and `0.023` for large ones (17.3x, COCO
  holdout).
* **At a matched low dose the local size gap vanishes.** When every object
  loses `0.02` of its attention, `Delta.02 = +0.000147 [+0.000009, +0.000286]`
  (DETR-R50) and `-0.000024 [-0.000294, +0.000238]` (Conditional DETR-R50),
  both inside the preregistered `+/-0.005` margin; H2 replicates this on
  VOC 2007 test.
* **The lesion budget changes a model comparison.** Under up to four cells, the
  difference between the small-object AP gaps of DETR-R50 and Conditional
  DETR-R50 changes by `+6.70 [+5.23, +8.00]` AP points between lesioning the
  third decoder layer and all six on COCO, `+9.64 [+6.26, +17.51]` on
  VOC 2007 test and `+7.76 [+5.15, +11.40]` on VOC 2012 val.
* **A relative budget reduces this depth dependence.** Removing half of each
  query's in-box attention reduces it by `T = +5.05 [+3.63, +6.69]` on COCO
  (post hoc), `+4.24 [-0.10, +14.04]` on VOC 2007 test (registered,
  inconclusive) and `+6.56 [+4.04, +10.05]` on VOC 2012 val (registered,
  confirmed).
* **Deformable detectors.** Four (head, sampling point) pairs remove nearly the
  same mass at both sizes (DEIMv2-S `0.039` vs `0.043` on VOC 2007 test, ratio
  0.91) and give small size interactions. A fixed 2x2-cell area removes 52x
  (DEIMv2-S) and 84x (RT-DETRv2) more mass from small objects and raises the
  size interaction to `+0.052` and `+0.029`. The fixed-area rule also changes
  which heads and sampling points are hit, so this comparison does not isolate
  dose.

## Installation

CPU-only, enough for `make tables`, `make verify` and `make test`:

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt
```

Full environment, needed for the GPU re-runs in `docs/REPRODUCE.md` (Python
3.12.3, torch 2.8.0+cu128, torchvision 0.23.0+cu128):

```
pip install -r requirements.txt
```

or, with conda:

```
conda env create -f environment.yml
conda activate size-conditioned-lesions
```

## Quick start: rebuild every table from the released records

```
make tables     # rewrites results/recomputed/*.json and prints every table
make verify     # recomputes and compares against the published numbers
make test       # unit tests for the dose, bootstrap and masking arithmetic
```

`make tables` needs only `numpy` and `scipy`; it reads the per-anchor records in
`records/` and the bootstrap arrays in `records/model_comparison/`.  Expect
about three minutes, most of it in the two 5,000-resample curve fits of Task R
and Task X.

## Data and checkpoints

Nothing that is large is stored in this repository.

* **Checkpoints.**  `scripts/download_checkpoints.py` fetches the seven models
  listed in `configs/models.yaml` and checks their SHA256.  Point
  `$CKPT_ROOT` at the directory that holds them; the default is `weights/`.
* **COCO.**  `scripts/prepare_coco_holdout.py` rebuilds the frozen split: the
  5,000 val2017 image ids permuted with PCG64 seed 20270925, positions
  1000-4999.  Point `$DATA_ROOT` at a COCO root containing `val2017/` and
  `annotations/instances_val2017.json`.
* **PASCAL VOC.**  `scripts/prepare_voc.py` converts VOC 2007 test (4,952
  images) and VOC 2012 val (5,823 images) to COCO format.  `difficult` objects
  become `iscrowd = 1`, so the official COCO evaluator ignores them; the 20 VOC
  classes are mapped onto their COCO category ids.

Every script resolves `$DATA_ROOT`, `$CKPT_ROOT`, `$OUT_ROOT` and `$WORK_ROOT`
from the environment and falls back to `data/`, `weights/`, `out/` and
`work/`, so the repository itself contains no absolute path.

## Full reproduction

`docs/REPRODUCE.md` gives the command, the hardware and the expected wall-clock
time for every GPU re-run, in dependency order.  The short version:

```
export DATA_ROOT=/path/to/data CKPT_ROOT=/path/to/weights
python experiments/local_quality/recompute_tables.py --json out/table1.json
python experiments/detector_ap/task_c_v2.py --model detr --layers 2
python experiments/deformable/task_n2_deim_official.py --data voc
```

Each GPU script writes one JSONL of per-anchor rows; the offline scripts in
`experiments/*/` consume those rows and never touch a model.

## Repository layout

```
configs/            model table (hashes, layer indices) and one YAML per analysis
lesions/            small importable helpers: dose.py, bootstrap.py, data/
experiments/        the frozen analysis scripts, one folder per paper section
scripts/            download, dataset preparation, table reproduction, verification
records/            per-anchor JSONL (gzip) and bootstrap arrays (.npz, .npy)
results/            one JSON per paper table, plus the per-draw aggregates
registrations/      the pre-registration texts, their hashes and the outcome list
docs/               reproduction guide, paper-to-code map, expected numbers
tests/              unit tests (no checkpoint, no GPU)
```

## Registered analyses

There are eighteen registered predictions.  Fifteen are confirmed, two are
contradicted and one is inconclusive; the full list with estimates and
intervals is in `registrations/OUTCOMES.md`, and the texts are in
`registrations/`.

| block | data | predictions | outcome |
|---|---|---|---|
| H1-H4 | VOC 2007 test | dense size interaction, low-dose equivalence, deformable null, detector-level gap | 4 confirmed |
| N1-N3 | VOC 2007 test | fixed-area dose ratio, fixed-area interaction, fixed-count null | 3 confirmed |
| D1-D3 | VOC 2007 test | Deformable DETR version of N1-N3 | 3 confirmed |
| R1-R3 | VOC 2007 test | relative dose explains more, low and high proportional dose | 3 confirmed |
| G1-G2 | VOC 2007 test | size neutrality of two deletion rules | 1 contradicted, 1 confirmed |
| G5 | VOC 2007 test | size neutrality of the dose-calibrated rule | contradicted |
| Y1-Y2 | VOC 2007 test, VOC 2012 val | the model-comparison reduction | 1 inconclusive, 1 confirmed |

`registrations/TIMELINE.md` records, for each registration, when it was frozen,
when the first run on the confirmatory data started, and what was already known
at freeze time.

## Paper-to-code map

`docs/PAPER_MAP.md` maps every table and figure to its result file, its record
file, the script that produced it and the command that reruns it.  A one-line
summary:

| paper item | script | rebuilt by |
|---|---|---|
| Table 1 | `experiments/local_quality/recompute_tables.py` | `make tables` |
| Table 3 (H1, H2) | same script, VOC records | `make tables` |
| Table 3 (H3) | `experiments/deformable/*` | GPU re-run |
| Table 3 (H4) | `experiments/detector_ap/task_c_v2.py` | GPU re-run |
| Table 3 (model comparison) | `experiments/model_comparison/recompute_y.py` | `make tables` |
| Task R table | `experiments/relative_dose/r_voc_analyze.py` | `make tables` |
| Task N table | `experiments/fixed_area/n_voc_analyze.py` | `make tables` |
| Table X | `experiments/transfer/task_x.py` | `make tables` |
| deletion-metric tables | `experiments/deletion_metrics/*` | GPU re-run |

## Known differences

* **Table 1 reproduces to 6e-5 - 2.3e-4, not to six decimals.**  The per-anchor
  records in this repository were regenerated for the release on a different
  software stack (Python 3.12.3, torch 2.8.0, numpy 2.3.2) from the one that
  produced the manuscript numbers.  `Delta4` moves by `+1.0e-4` for DETR-R50
  and `+2.1e-4` for Conditional DETR-R50; the bootstrap interval endpoints move
  by up to `1.8e-3`.  `Delta.02` moves by less than `1e-6`.  The DETR-R50 row
  of Table 3 (VOC 2007 test) moves by `+1.9e-4` for `Delta4` and `-4.9e-5` for
  `Delta.02`; the Conditional DETR-R50 row moves by less than `7e-6`.  `make
  verify` therefore allows `5e-4` on the point estimates of Table 1 and
  Table 3 and `2.5e-3` on the Table 1 interval endpoints, prints the deviation
  for every row, and is exact (`1e-6` or `1e-3`) for the offline contrasts and
  for every block read from a result file.
* **The `grad` saliency map is not bitwise reproducible on GPU.**  Replaying
  the same images moves its AUC by up to `0.0159`; the `random` and `attn` maps
  replay byte-identically.  Every reported `grad` number is subject to that
  tolerance and is never used in a registered test.
* **Per-image detector outputs are not included.**  The H3, H4, D and G blocks
  need detection files of several hundred megabytes, so the release ships the
  aggregate JSON in `results/` instead of the predictions.  `docs/EXPECTED.md`
  states for each block whether `make verify` recomputes it from records or
  reads it from a result file.

## License

MIT, see `LICENSE`.

This repository is prepared for double-blind review.  It contains no author
names, affiliations, machine names or absolute paths; the check used to verify
that is described in `docs/EXPECTED.md`.
