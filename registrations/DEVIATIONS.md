# Deviations from the registrations and from the announced protocol

Everything below is a departure that a reader should know about.  None of it
changes a registered threshold, seed or decision rule; where a departure could
have changed a verdict, the verdict is taken from the run that follows the
registration.

## 1. DEIMv2-S ran with the wrong decoder configuration once

The registered model uses the official decoder configuration,
`interaction_indexes = [3, 7, 11]`.  The first VOC execution of Task N read a
configuration file that specified `[5, 8, 11]`.  The mismatch was found by
replaying two VOC images and comparing the clean `Q` against the stored rows;
the run that produced the stored rows matched `[5, 8, 11]` bit for bit.

The whole DEIMv2-S part of Task N was re-executed under the registered
configuration, with the code, seeds, anchor rules, bootstrap and verdict rules
unchanged.  Only the configuration path differs.  The re-execution is the one
that carries N1-N3, and it is also the source of every DEIMv2-S number quoted
in the paper.  The original run is kept and reported next to it; nothing was
deleted or overwritten.  The other DEIMv2-S analyses were audited the same way
and were already on the official configuration.

## 2. Task G started fifteen minutes early

The addendum was frozen at 2026-11-17 15:20 and the two shards were launched at
15:21; the explicit authorisation arrived at 15:36.  No Task G statistic on a
VOC image existed at hash time, and no output of the 15:21 launch was read
before the authorisation.  The confirmatory ordering is therefore intact, but
the launch is recorded here and in `TIMELINE.md`.

## 3. The 25 % box-cell arm was not run on VOC

The registered VOC run of Task R covers the fixed four-cell arm, the seven
equal-mass arms and the four proportional arms.  The `fraction_inside` arm
(25 % of the box's cells by count) exists only on the COCO holdout, because the
VOC registration was written after the COCO analysis and did not ask for it.
No VOC conclusion depends on it.

## 4. `grad` saliency is not bitwise reproducible

The gradient map depends on non-deterministic GPU reductions.  Replaying the
same images moves its AUC by up to `0.0159` and its per-step score by up to
`0.19`.  The `random` and `attn` maps replay byte-identically, and every
registered deletion-metric test uses the `random` map.  All `grad` numbers are
reported with this tolerance and are descriptive.

## 5. Region geometry in Task N

The addendum describes the lesion region as "2x2 / 3x3 stride-16 feature
cells".  The frozen code sizes the region from the model's own preprocessing
scale, which for DEIMv2-S and RT-DETRv2 makes it a 32x32 / 48x48 pixel square
in the original image, and for Deformable DETR a genuine stride-16 cell square
in the processor input frame.  An earlier internal note claimed the code and
the text disagreed; that note was wrong and was retracted.  The verdicts follow
the frozen code, which is what the addendum describes, and the region edge
lengths are reported next to the results.

## 6. Task G4's development criterion was not met, and the axis was stopped

Before looking at the G4 development results, a success criterion was fixed:
the small-minus-large median AUC had to lie inside `[-0.05, +0.05]` for both
the random and the gradient map.  The random map met it (`+0.0136
[-0.0074, +0.0372]`); the gradient map did not (`-0.2326
[-0.2548, -0.2071]`).  The axis was stopped at that point and no G4 variant was
run.  The subsequent G5 registration concerns the random map only and discloses
this failure in its own text; the failure is repeated in
`results/tableG5_voc.json` and in the paper.

## 7. The per-anchor records were regenerated for the release

The records shipped here were produced on Python 3.12.3 / torch 2.8.0 /
numpy 2.3.2.  They reproduce the manuscript's Table 1 `Delta4` to
`5.6e-5` (DETR-R50) and `2.3e-4` (Conditional DETR-R50) and its interval
endpoints to `1.8e-3`; `Delta.02` moves by less than `1e-6`.  Every other table
reproduces exactly.  See "Known differences" in `README.md`.

## 8. Per-image detector outputs are not shipped

The H3, H4, D and G blocks consume per-image detection files of several hundred
megabytes.  The release ships the aggregate JSON that those files produced
(`results/tableH3_voc.json`, `tableH4_voc.json`, `tableD_voc.json`,
`tableG_voc.json`, `tableG5_voc.json`) and the per-shard summaries in
`records/detector_ap/`.  `docs/EXPECTED.md` marks these blocks as read from a
result file rather than recomputed from records.

## 9. Three reports were re-issued with an appended erratum

Two Task N reports and one Task T report gained an appended erratum after the
DEIMv2-S configuration was corrected.  The original text was not edited; the
erratum states that the current DEIMv2-S numbers are the re-executed ones.
