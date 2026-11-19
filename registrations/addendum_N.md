# Confirmatory Registration v1 — Addendum N: within-model dose manipulation (ICLR 55966)

Status: written after the COCO development runs of Task N (RESULTS_N_dev.md and RESULTS_N_dev_v2.md) and before any Task N computation on PASCAL VOC 2007 test. Hypotheses were formed on the development runs and are confirmatory only on VOC 2007 test.

## Data and models
PASCAL VOC 2007 test (all 4,952 images), with the data, class map, difficult-object rule, size bins and anchor rule of CONFIRM_REG_v1. DEIMv2-S (official configuration) and RT-DETRv2-R50vd; zero-based decoder layer 2; query-specific lesions on (head, sampling point) pairs; dose = removed mean-head attention mass.

## Lesions
(i) four in-box pairs (as in CONFIRM_REG_v1, H3); (ii) equal mass 0.02; (iii) fixed region 2x2: all pairs whose sampling location falls inside a square of 2x2 stride-16 feature cells centred on the target box centre; (iv) fixed region 3x3, likewise. Primary handling: a head whose pairs are all removed outputs zero; anchors whose region contains no pair are kept with dose 0. The code is frozen at the SHA256 recorded in decision_log.md with this addendum.

## Statistic
Δ_lesion = E[Q_clean − Q_lesion | small] − E[Q_clean − Q_lesion | large], 95% percentile interval from 5,000 whole-image bootstrap resamples, PCG64 seed 20271001; token-selection seed 20270930.

## Hypotheses (each model separately)
N1 A fixed-area lesion delivers a larger dose to small objects: mean removed mass small / large > 5 for the 2x2 region in both models.
N2 A fixed-area lesion produces a size interaction: lower 95% bound of Δ_2x2 > +0.005 in both models.
N3 A fixed-count lesion on the same models does not: 95% interval of Δ_four-pairs within [−0.005, +0.005] in both models.

Outcome per hypothesis: "confirmed" if the rule holds in both models; "contradicted" if in either model the interval lies wholly on the other side of the boundary (for N1: ratio ≤ 1 in either model); otherwise "inconclusive". The 3x3 region, the secondary handling (definable anchors only), counts of zeroed heads and zero-dose anchors are reported descriptively. All outcomes are reported regardless of result. Nothing may change after this text is hashed.
