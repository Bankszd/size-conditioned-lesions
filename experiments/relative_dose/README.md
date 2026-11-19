# Relative dose (Task R)

`task_r.py` runs the proportional lesions on GPU (fraction `f` of each anchor's
own clean in-box mass) alongside the fixed-four and equal-mass arms.
`r_voc_analyze.py` is the offline analysis: it bins every equal-mass row by
`r = removed mass / clean in-box mass`, fits a piecewise-linear curve through
the bin means plus the origin, and uses that curve to predict the fixed
four-cell size gap.  The absolute-dose curve is fitted the same way on the
named doses.

The explained share is `predicted gap / measured gap`; R1 compares the relative
and absolute curves, R2 and R3 look at `f = 0.10` and `f = 0.75`.

    python experiments/relative_dose/r_voc_analyze.py \
        records/relative_dose/r_voc_detr.jsonl.gz \
        records/relative_dose/r_voc_conditional_detr.jsonl.gz out/tableR.json
