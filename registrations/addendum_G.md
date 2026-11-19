# Confirmatory Registration v1 — Addendum G: size neutrality of deletion metrics (ICLR 55966)

Status: written after the COCO development run of Task G (RESULTS_G_dev.md, SHA256 aa117f025b60320c6f14b54615c89f170a4a0d148765a812cde1aa406a7810d5) and before any Task G computation on PASCAL VOC 2007 test. The hypotheses below were formed on that development run and are confirmatory only on VOC 2007 test.

## Data and models
- PASCAL VOC 2007 test, all 4,952 images, with the data, class map, difficult-object rule and size bins of CONFIRM_REG_v1.
- DETR-R50 and RT-DETRv2-R50vd, the same checkpoints and native evaluation transforms as in the development run.
- Targets: at most one detector-matched ground-truth object per image and size bin, smallest annotation id on ties; targets with clean score 0 are excluded and counted.

## Procedure
Identical to the development run: code task_g.py (DETR-R50) and task_g_rt.py (RT-DETRv2), frozen at the SHA256 values recorded in decision_log.md together with this addendum. Saliency maps random, attn and grad on the 16x16-pixel patch grid of the model input; deletion in descending saliency order by multiplying patch pixels by (1 - alpha) in normalized space; two step rules, 50 steps each: fixed (4 patches per step) and relative (5% of the target box's patch count per step); score = max over same-class predictions of p x IoU, normalized by its clean value; AUC = mean normalized score over steps 1..50. Random-saliency seeds as in the development run.

## Primary statistic
For each model and step rule: dAUC = median AUC(small) - median AUC(large) under the random saliency map. Medians are used because the development run showed that the mean AUC is dominated by targets with near-zero clean scores. 95% percentile interval from 2,000 whole-image bootstrap resamples, PCG64 seed 20271001.

## Hypotheses (each model evaluated separately)
G1 Under the fixed step rule, the random-saliency AUC depends on object size with small objects lower: upper 95% bound of dAUC(fixed) < 0 in both models.
G2 Under the relative step rule, the random-saliency AUC depends on object size with small objects higher: lower 95% bound of dAUC(relative) > 0 in both models.

Outcome for each hypothesis: "confirmed" if the rule holds in both models; "contradicted" if, in either model, the interval lies wholly on the opposite side of zero; otherwise "inconclusive". Both outcomes, all intervals and all target counts are reported regardless of result.

## Descriptive outputs (not tested)
Per model, saliency map, step rule and size bin: mean, median and 99%-winsorized mean AUC with intervals; the fraction of in-box patches actually deleted after steps 1, 5, 10, 20, 30, 40, 50; the ranking of the three saliency maps by median AUC. The grad map is not bitwise reproducible on GPU (development replay: max |dAUC| 0.0058); its results are reported with this tolerance.

No hypothesis, statistic, step rule, seed or data rule may change after this text is hashed.
