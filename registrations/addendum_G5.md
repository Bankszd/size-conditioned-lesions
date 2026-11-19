# Confirmatory Registration v1 — Addendum G5: an attention-dose deletion rule (ICLR 55966)

Status: written after the COCO development run of Task G4 (RESULTS_G4_dev.md, SHA256 35c1bfe4e019bd66fa32ed9377f7f444351282203c45523218a8996bd200f1d2) and before any G4-rule computation on PASCAL VOC 2007 test.

Disclosure. On the development set we had fixed, before seeing results, a success criterion requiring the small-minus-large median AUC to lie within [-0.05, +0.05] for both the random and the gradient saliency maps. The random map met it (+0.0136 [-0.0074, +0.0372]); the gradient map did not (-0.2326 [-0.2548, -0.2071]). We now consider that criterion mis-specified: size neutrality of a deletion metric means that an uninformative (random) saliency map scores the same at every object size, which is also how hypotheses G1 and G2 were defined; an informative map may legitimately differ in faithfulness across sizes. The hypothesis below therefore concerns the random map only. The development failure under the original criterion will be reported alongside the outcome.

## Data and models
PASCAL VOC 2007 test (all 4,952 images), with the data, class map, difficult-object rule, size bins and target rule of Addendum G. DETR-R50 and RT-DETRv2-R50vd, the checkpoints and native transforms of Addendum G.

## Deletion rule (G4 rule)
Dose weights: the matched query's clean head-mean cross-attention in the last decoder layer, mapped to the 16-pixel patch grid of the model input over the whole image and normalized to sum to one. For DETR-R50, the mapping of RESULTS_G4_dev.md; for RT-DETRv2-R50vd, the mean-head attention weight of each sampling point, summed into the patch that contains its sampling location, over all feature levels. Patches are deleted over the whole image in descending saliency order; step k deletes until the cumulative dose reaches 0.02k (k = 1..50), with the crossing patch blended at pixel level. Deletion, score, AUC (mean over steps 1..50) and saliency maps (random, grad, attn) as in Addendum G. Code frozen at the SHA256 recorded in decision_log.md with this addendum.

## Statistic
dAUC = median AUC(small) - median AUC(large) under the random saliency map; 95% percentile interval from 2,000 whole-image bootstrap resamples, PCG64 seed 20271001.

## Hypothesis (each model separately)
G5 Under the G4 rule, the random-saliency AUC is size-neutral: the 95% interval of dAUC lies within [-0.05, +0.05] in both models.

Outcome: "confirmed" if the rule holds in both models; "contradicted" if, in either model, the interval lies wholly outside [-0.05, +0.05]; otherwise "inconclusive". Reported descriptively: dAUC for grad and attn (attn shares the dose weights and is circular), per-size median AUC, the in-box dose and patch fractions per step, and, from the Addendum G runs on the same images, the corresponding standard-rule values for comparison. All outcomes are reported regardless of result. Nothing may change after this text is hashed.
