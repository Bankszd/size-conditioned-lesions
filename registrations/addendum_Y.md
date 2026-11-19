# Confirmatory Registration v1 — Addendum Y (VOC 2007 test): lesion rule and model comparison across depth

Data: PASCAL VOC 2007 test, all 4,952 images; VOC->COCO class map; difficult objects as crowd; COCO size thresholds; as in H4.
Models: DETR-R50 and Conditional DETR-R50 (same checkpoints as H4).
Readout: detector AP; every query is lesioned inside its own clean predicted box. S = dAP_small - dAP_large (AP points), dAP = AP_lesion - AP_clean.
Depths: L3 (zero-based decoder layer 2) and ALL (layers 0-5; each layer's dose computed from its own clean attention).
Rules: four (up to four in-box cells, as in H4); prop50 (remove 0.50 of the query's clean head-mean in-box mass at each lesioned layer, seeded random cell order, last cell fractional); geo50 (delete floor(0.5 n) of the n in-box cells in the same seeded order, and if n is odd attenuate the next cell by a factor 0.5; n = 0: no deletion).
Statistics: D = S_DETR - S_Cond for each depth and rule; I_rule = D_ALL - D_L3; primary T = I_four - I_prop50. One shared 2,000 x 4,952 whole-image resample matrix, PCG64 seed 20271001, used for all models, depths and rules; percentile 95% intervals.
Known before freezing: H4 (four and mass 0.02 at L3 on VOC); COCO Task Y (four, mass 0.02, prop50 at L3 and ALL). Not known: any VOC result at depth ALL, any VOC prop50 or geo50 result, any COCO geo50 result.
Prediction Y1: the lower 95% bound of T is > 0, i.e. confirmed. Contradicted if the upper bound is < 0; inconclusive otherwise.
Descriptive, no prediction: I_four, I_prop50, I_geo50, T_g = I_geo50 - I_prop50; AP, AP_S, AP_L and their changes for every arm; removed in-box mass fractions.
All outcomes are reported.