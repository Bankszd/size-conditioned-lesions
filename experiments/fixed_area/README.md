# Fixed-area lesions (Task N)

`n_voc_analyze.py` turns the per-anchor rows of a deformable run into the
N1-N3 quantities:

* N1 - mean removed mass small / large for the 2x2 region;
* N2 - `Delta_2x2 = E[L|small] - E[L|large]`, `L = Q_clean - Q_lesion`;
* N3 - the same contrast for the fixed four-pair arm.

Primary handling: a head whose twelve sampling points all fall inside the
lesion region outputs zero, and an anchor whose region contains no sampling
point is kept with dose 0.  The supplementary handling drops those anchors
instead; both are reported.

    python experiments/fixed_area/n_voc_analyze.py \
        --deim records/deformable/task_d2_n2_deim_voc_officialcfg.jsonl.gz \
        --out out/tableN.json
