# Confirmatory Registration v1 — Addendum R: relative dose (ICLR 55966)

Status: written after the COCO analyses RESULTS_R0.md (SHA256 ee1f905d8d88964264714090b2af5f881ae21d26e4e427c4d1cda5730c15d1a4) and RESULTS_R1_dev.md (SHA256 dbb7765e8e94189d14734ba972320d02bea104efae998a20611291476aa19914), and before any Task R computation on PASCAL VOC 2007 test. Hypotheses were formed on those analyses and are confirmatory only on VOC 2007 test.

## Data and models
PASCAL VOC 2007 test (all 4,952 images), with the data, class map, difficult-object rule, size bins and anchor rule of CONFIRM_REG_v1. DETR-R50 and Conditional DETR-R50; zero-based decoder layer 2; all-query lesions; eligibility as for the fixed four-cell lesion (at least four in-box and four out-of-box cells); token-selection seed 20270930. Code frozen at the SHA256 recorded in decision_log.md with this addendum.

## Lesions
Fixed four in-box cells; equal mass m in {0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80} (anchors whose clean in-box mass is below m are excluded from that arm); proportional lesions removing a fraction f in {0.10, 0.25, 0.50, 0.75} of each anchor's clean in-box attention mass, deleted as in the equal-mass lesion.

## Quantities
For every anchor and arm: loss L = Q_clean − Q_lesion; relative dose r = removed mass / clean in-box mass. Size gap of an arm: Δ = E[L | small] − E[L | large].
Relative-dose curve: pooling all equal-mass rows over all sizes, the mean L in ten equal-width bins of r ([0,0.1), …, [0.9,1.0]) is placed at each bin's mean r; with the node (0,0), the curve is piecewise linear; r above the last node takes the last node's value.
Absolute-dose curve: pooling all equal-mass rows over all sizes, mean L at each m with the node (0,0), piecewise linear, extrapolated linearly beyond 0.80 from the last segment.
Explained share: the predicted size gap of the fixed four-cell arm (each anchor's loss read from the curve at its own dose, then averaged by size) divided by the measured gap; computed for both curves.
Intervals: 5,000 whole-image percentile bootstrap resamples, PCG64 seed 20271001; both curves are refitted within every resample, and the difference of the two explained shares is computed within each resample.

## Hypotheses (each model separately)
R1 The relative-dose curve explains more of the fixed four-cell size gap than the absolute-dose curve: lower 95% bound of (share_relative − share_absolute) > 0, in both models.
R2 At a low proportional dose there is no size gap: the 95% interval of Δ for f = 0.10 lies within [−0.005, +0.005], in both models.
R3 At a high proportional dose a size gap remains: lower 95% bound of Δ for f = 0.75 > 0, in both models.

Outcome per hypothesis: "confirmed" if the rule holds in both models; "contradicted" if, in either model, the interval lies wholly on the other side of the boundary (R1, R3: upper bound < 0; R2: interval wholly outside the margin); otherwise "inconclusive". VOC 2007 test contains few small objects (about 167 eligible), so R3 may be inconclusive for lack of power; this was anticipated when writing this text. All outcomes, both explained shares with intervals, all Δ for every f and m, and per-size counts are reported regardless of result. Nothing may change after this text is hashed.
