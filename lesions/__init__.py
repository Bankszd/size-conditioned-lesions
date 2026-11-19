"""Small, dependency-light helpers used across the released analyses.

The functions here are re-usable implementations of the dose bookkeeping that
the frozen experiment scripts perform.  They are provided so that a reader can
write new lesions on top of the paper's definitions, and so that the arithmetic
is unit-tested in ``tests/``.

Important: every number reported in the paper is produced by the frozen
scripts under ``experiments/`` and not by this package.  See
``docs/PAPER_MAP.md``.
"""

from .dose import (  # noqa: F401
    apply_logit_mask,
    box_cell_mask,
    equal_mass_select,
    fixed_area_mask,
    fixed_count_select,
    cell_fraction_select,
)
from .bootstrap import (  # noqa: F401
    draw_matrix,
    image_weights,
    size_gap,
    percentile_interval,
)

__all__ = [
    "apply_logit_mask",
    "box_cell_mask",
    "equal_mass_select",
    "fixed_area_mask",
    "fixed_count_select",
    "cell_fraction_select",
    "draw_matrix",
    "image_weights",
    "size_gap",
    "percentile_interval",
]
