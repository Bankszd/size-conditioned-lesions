# Deletion metrics (G, G2, G3, G4, G5)

`task_g5.py` (DETR-R50) and `task_g5_rt.py` (RT-DETRv2-R50vd) compute deletion
curves on the 16-pixel patch grid of the model input.

Three saliency maps: `random`, `grad` (absolute input gradient of the matched
query's class score) and `attn` (the matched query's head-averaged
cross-attention).  Two step rules: a fixed number of patches per step, and a
relative rule that deletes a fixed fraction of the target box's patches.

The G4/G5 rule instead deletes patches over the whole image in descending
saliency order until the cumulative *attention dose* reaches `0.02 k`.  Because
`attn` shares the dose weights, its G5 numbers are circular and reported only
as a description.

The registered statistic is `dAUC = median AUC(small) - median AUC(large)`
under the random map, with a 2,000-draw whole-image bootstrap.

    python experiments/deletion_metrics/task_g5.py --data voc --model detr \
        --cache $DATA_ROOT/voc2007test --out out/g5.jsonl.gz

`merge_g5.py` merges contiguous shards and `g5_dose.py` checks the dose ledger.
The `grad` map is not bitwise reproducible on GPU; the replay tolerance is
recorded in `registrations/DEVIATIONS.md`.
