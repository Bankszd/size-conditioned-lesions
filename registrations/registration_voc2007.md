# Confirmatory Registration v1 — Size-conditioned spatial lesions (ICLR 55966 rebuttal)

Status: written after all COCO results reported in the rebuttal; written before any model is run
on the confirmation set. Hypotheses H3–H4 were formed after observing the COCO 4,000-image holdout
and are therefore confirmatory only on this new set.

## Data
- PASCAL VOC 2007 test, all 4,952 images; non-difficult objects only. "difficult" objects are
  treated like COCO crowd annotations (excluded as anchors, ignored in AP).
- VOC→COCO class map: aeroplane→airplane, bicycle→bicycle, bird→bird, boat→boat, bottle→bottle,
  bus→bus, car→car, cat→cat, chair→chair, cow→cow, diningtable→dining table, dog→dog,
  horse→horse, motorbike→motorcycle, person→person, pottedplant→potted plant, sheep→sheep,
  sofa→couch, train→train, tvmonitor→tv. Predictions of the other 60 COCO classes are discarded
  before matching and evaluation.
- Size bins use COCO thresholds (32², 96²) on box area w×h (VOC has no mask area).
- Each model uses its native evaluation transform, unchanged from the COCO experiments.

## Models and intervention
- Dense: DETR-R50, Conditional DETR-R50 (same checkpoints as the paper).
  Deformable: DEIMv2-S (official config, interaction_indexes=[3,7,11]), RT-DETRv2-R50vd.
- Decoder cross-attention, zero-based layer index 2. Lesion, anchor, eligibility, dose, local
  quality Q, D4, L_m, Δ4 and Δ.02 are defined exactly as in the paper (Sections 2–3) and, for
  deformable models, as in RESULTS_D2 (unit = (head, sampling point) pair, query-specific).
- Token-selection seed 20270930. Whole-image percentile bootstrap, 5,000 resamples,
  PCG64 seed 20271001.
- Detector-level readout (dense models only): the per-query predicted-box protocol of RESULTS_C,
  layer index 2 only; S (pp) = ΔAPs − ΔAPl in COCO-style percentage points, computed with the
  official COCO evaluator on the converted annotations; 2,000 paired whole-image bootstrap resamples.

## Hypotheses and decision rules (each model evaluated separately)
H1 Dense fixed-count interaction persists: lower 95% bound of Δ4 > +0.005 in both dense models.
H2 Dense low-dose equivalence: 95% interval of Δ.02 within [−0.005, +0.005] in both dense models.
H3 No fixed-count interaction in deformable models: 95% interval of Δ4 within [−0.005, +0.005]
   in both deformable models.
H4 Detector-level size gap under fixed count (dense): upper 95% bound of S(in4) < 0 in both
   dense models; and the 95% interval of S(in_m0.02) within [−0.5, +0.5] pp in both dense models.
   (The ±0.5 pp margin is set here without external validation and is descriptive of practical size.)

For each hypothesis: "confirmed" if the rule holds in every listed model; "contradicted" if, in any
listed model, the interval lies wholly on the opposite side of the stated boundary (e.g. Δ4 upper
bound < +0.005 for H1; interval wholly outside the margin for H2–H3); otherwise "inconclusive".
All four hypotheses, all intervals and all eligible counts are reported regardless of outcome.
No hypothesis, margin, seed, layer or data rule may change after this text is hashed.
A deletion-metric experiment (Task G) may be registered later as a separate addendum, before it
is run on this set.