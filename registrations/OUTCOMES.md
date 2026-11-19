# Registered predictions and outcomes

Eighteen predictions were registered before their data were touched.  Fifteen
are confirmed, two are contradicted and one is inconclusive.  The verdict of
each one follows the rule written in its own registration; no margin, seed or
data rule was changed after the text was hashed.

## Dense detectors on VOC 2007 test (H1-H4)

| id | prediction | estimate [95% interval] | outcome |
|---|---|---|---|
| H1 | the fixed-count size interaction persists: lower bound of Delta4 > +0.005 in both dense models | DETR-R50 `+0.028481`, Conditional `+0.033838`; lower bounds `+0.021822` / `+0.026036` | confirmed |
| H2 | the low-dose size gap vanishes: interval of Delta.02 inside +/-0.005 in both models | DETR-R50 `+0.000330 [+0.000122, +0.000471]`, Conditional `-0.000285 [-0.001281, +0.000204]` | confirmed |
| H3 | deformable detectors show no fixed-count interaction: interval of Delta4 inside +/-0.005 in both | DEIMv2-S `-0.000498 [-0.001375, +0.000378]`, RT-DETRv2 `+0.001427 [-0.000725, +0.004418]` | confirmed |
| H4 | detector-level size gap under a fixed count: upper bound of S(in4) < 0 in both dense models, and S(in_m0.02) inside +/-0.5 pp | S(in4) `-2.445 [-3.625, -1.123]` / `-5.392 [-7.898, -1.374]`; S(in_m0.02) `+0.021 [-0.268, +0.191]` / `-0.066 [-0.209, +0.061]` | confirmed |

## Fixed-area lesions on VOC 2007 test (N1-N3)

| id | prediction | estimate [95% interval] | outcome |
|---|---|---|---|
| N1 | a 2x2 region delivers more mass to small objects: ratio > 5 in both deformable models | `52.27` (DEIMv2-S), `84.32` (RT-DETRv2) | confirmed |
| N2 | the 2x2 region produces a size interaction: lower bound of Delta_2x2 > +0.005 in both | `+0.052413 [+0.045329, +0.059467]`, `+0.029226 [+0.021221, +0.038315]` | confirmed |
| N3 | the fixed four-pair lesion does not: interval of Delta_four inside +/-0.005 in both | `-0.000298 [-0.000789, +0.000185]`, `+0.001313 [-0.000345, +0.004296]` | confirmed |

## Deformable DETR-R50 on VOC 2007 test (D1-D3)

| id | prediction | estimate [95% interval] | outcome |
|---|---|---|---|
| D1 | 2x2 mass ratio small/large > 2 | `0.495824 / 0.068339 = 7.2554` | confirmed |
| D2 | lower bound of Delta_2x2 > 0 | `+0.010496 [+0.008318, +0.012746]` | confirmed |
| D3 | four-pair and equal-mass intervals inside +/-0.005 | `+0.000268 [-0.000414, +0.000941]`, `+0.000136 [-0.000004, +0.000310]` | confirmed |

## Relative dose on VOC 2007 test (R1-R3)

| id | prediction | estimate [95% interval] | outcome |
|---|---|---|---|
| R1 | the relative-dose curve explains more of the four-cell gap than the absolute-dose curve: lower bound of the difference > 0 in both | `+0.2928 [+0.2126, +0.4277]` (DETR-R50), `+0.6398 [+0.4944, +0.8490]` (Conditional) | confirmed |
| R2 | no size gap at `f = 0.10`: interval of Delta inside +/-0.005 in both | `[+0.000386, +0.001386]`, `[-0.002175, +0.001952]` | confirmed |
| R3 | a gap remains at `f = 0.75`: lower bound > 0 in both | `+0.001265`, `+0.000479` | confirmed |

## Deletion metrics on VOC 2007 test (G1, G2, G5)

| id | prediction | estimate [95% interval] | outcome |
|---|---|---|---|
| G1 | under the fixed rule the random-map AUC is lower for small objects: upper bound of dAUC < 0 in both models | `+0.0061 [+0.0001, +0.0116]` (DETR-R50), `+0.0107 [-0.0037, +0.0208]` (RT-DETRv2) | contradicted (the DETR interval lies wholly above zero) |
| G2 | under the relative rule the random-map AUC is higher for small objects: lower bound > 0 in both | `+0.8266 [+0.8185, +0.8373]`, `+0.7094 [+0.6944, +0.7223]` | confirmed |
| G5 | under the dose-calibrated rule the random-map AUC is size-neutral: interval of dAUC inside [-0.05, +0.05] in both | `+0.1288 [+0.1083, +0.1484]`, `+0.0769 [+0.0490, +0.0935]` | contradicted |

## Model-comparison reduction (Y1, Y2)

| id | prediction | estimate [95% interval] | outcome |
|---|---|---|---|
| Y1 | on VOC 2007 test the lower bound of `T = I(four) - I(prop50)` is > 0 | `+4.236 [-0.099, +14.043]` AP points | inconclusive |
| Y2 | on VOC 2012 val the lower bound of `T` is > 0 | `+6.560 [+4.040, +10.050]` AP points | confirmed |

The same contrast on the COCO holdout, `+5.05 [+3.63, +6.69]`, was not a
registered prediction: it was measured during the development of the
reduction and is reported as a description.
