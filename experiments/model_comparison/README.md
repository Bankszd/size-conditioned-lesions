# Model comparison across depth (D, I, T)

`task_y_paired.py` computes, for one image resample, the AP / AP_S / AP_L of
every arm and writes the per-draw arrays to an `.npz`.  `recompute_y.py` turns
those arrays into the contrasts of Table 3 without touching a GPU:

    D(rule, depth)   = S(DETR) - S(Conditional)
    I(rule)          = D(rule, all layers) - D(rule, layer 2)
    T                = I(four cells) - I(prop50)

All arms share one resample matrix (`records/model_comparison/task_y_draws_2000.npy`,
2,000 x 4,000) so the contrasts are formed draw by draw.

    python experiments/model_comparison/recompute_y.py \
        --npz records/model_comparison/task_y_boot_geo_detr.npz \
              records/model_comparison/task_y_boot_geo_conditional_detr.npz

`task_y_geo.py` produced the `geo50` arm; `task_y_stats.py` is the original
aggregator and is kept for reference.
