# Confirmatory Registration v1 — Addendum D3: Deformable DETR (ICLR 55966)

Status: written after the COCO development run of Task D3 (RESULTS_D3_dev.md, SHA256 47d7ac1ab46395174f7590450fd3b4c69c3ebadc49cf7a99d461356e140751b4) and before any Task D3 computation on PASCAL VOC 2007 test. Hypotheses were formed on the development run and are confirmatory only on VOC 2007 test.

## Data and model
PASCAL VOC 2007 test (all 4,952 images), with the data, class map, difficult-object rule, size bins and anchor rule of CONFIRM_REG_v1. Deformable DETR-R50 (HuggingFace SenseTime/deformable-detr, model.safetensors SHA256 caf1e3e61283c6ce35cd2d9adaa7033cf40997d4dfe434003bcdb9085cc8cf9b); zero-based decoder layer 2; query-specific lesions on (head, sampling point) pairs over all feature levels; dose = removed mean-head attention mass.

## Lesions
(i) four in-box pairs; (ii) equal mass 0.02; (iii) fixed region 2x2 and (iv) 3x3 stride-16 cells centred on the target box centre, with the primary handling of Addendum N (a fully covered head outputs zero; an empty region has dose 0). Code frozen at the SHA256 recorded in decision_log.md with this addendum.

## Statistic
Δ_lesion = E[Q_clean − Q_lesion | small] − E[Q_clean − Q_lesion | large], 95% percentile interval from 5,000 whole-image bootstrap resamples, PCG64 seed 20271001; token-selection seed 20270930.

## Hypotheses
D1 A fixed-area lesion delivers a larger dose to small objects: mean removed mass small / large > 2 for the 2x2 region.
D2 A fixed-area lesion produces a size interaction: lower 95% bound of Δ_2x2 > 0.
D3 Four pairs and equal mass do not: the 95% intervals of Δ_four-pairs and Δ_mass0.02 lie within [−0.005, +0.005].

Outcome per hypothesis: "confirmed" if the rule holds; "contradicted" if the interval lies wholly on the other side of the boundary (for D1: ratio ≤ 1); otherwise "inconclusive". The 3x3 region, the supplementary handling and counts of zeroed heads and zero-dose anchors are reported descriptively. All outcomes are reported regardless of result. Nothing may change after this text is hashed.
