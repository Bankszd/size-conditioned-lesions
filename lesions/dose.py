"""Dose definitions and the four lesion rules of the paper.

Terminology used throughout (paper Section 2):

    cell      one spatial position of the decoder cross-attention map
    mass      head-averaged attention weight of a query on a cell
    in-box    cells whose feature-grid centre falls inside the ground-truth box
    Q         local quality, p(class) * IoU(predicted box, ground-truth box)
    m         dose, the attention mass removed by a lesion

Nothing in this module touches a model; it is pure array arithmetic.
"""

import numpy as np


def box_cell_mask(centres_xy, box_xyxy):
    """Boolean mask of the cells whose centre lies inside ``box_xyxy``.

    ``centres_xy`` is (n, 2) in feature-grid coordinates and ``box_xyxy`` is
    (x0, y0, x1, y1) in the same coordinate frame.
    """
    c = np.asarray(centres_xy, dtype=float).reshape(-1, 2)
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    return (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1)


def fixed_count_select(n_cells, k, rng):
    """Fixed-count lesion: ``min(k, n_cells)`` cells in random order."""
    n_cells = int(n_cells)
    k = int(min(k, n_cells))
    if n_cells <= 0:
        return np.zeros(0, dtype=int)
    return np.asarray(rng.permutation(n_cells)[:k], dtype=int)


def equal_mass_select(masses, target, order=None):
    """Equal-mass lesion.

    Whole cells are removed in ``order`` until adding the next one would
    overshoot ``target``; that next cell is then removed fractionally.  If the
    cells together hold less than ``target`` every cell is removed.

    Returns ``(indices, fractions)`` with ``fractions`` in ``(0, 1]``.  The
    realised dose differs from ``target`` by at most the mass of one cell's
    fraction, which the caller can check.
    """
    masses = np.asarray(masses, dtype=float).ravel()
    n = masses.size
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    if order is None:
        order = np.arange(n)
    order = np.asarray(order, dtype=int)

    total = float(masses.sum())
    if total <= float(target):
        return order, np.ones(n, dtype=float)

    idx, frac = [], []
    running = 0.0
    for cell in order:
        w = float(masses[cell])
        if running + w <= target + 1e-12:
            idx.append(cell)
            frac.append(1.0)
            running += w
            continue
        need = float(target) - running
        if need > 0.0 and w > 0.0:
            idx.append(cell)
            frac.append(min(1.0, need / w))
        break
    return np.asarray(idx, dtype=int), np.asarray(frac, dtype=float)


def cell_fraction_select(n_cells, fraction, order=None):
    """Cell-fraction lesion: ``floor(fraction * n)`` whole cells, plus half of
    the next cell when ``n`` is odd and ``fraction == 0.5`` style remainders
    appear.  Used by the geo50 arm."""
    n = int(n_cells)
    if n <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    if order is None:
        order = np.arange(n)
    order = np.asarray(order, dtype=int)

    exact = float(fraction) * n
    whole = int(np.floor(exact))
    rest = exact - whole
    idx = list(order[:whole])
    frac = [1.0] * whole
    if rest > 1e-12 and whole < n:
        idx.append(order[whole])
        frac.append(float(rest))
    return np.asarray(idx, dtype=int), np.asarray(frac, dtype=float)


def fixed_area_mask(centres_xy, centre, side_cells, stride):
    """Fixed-area lesion: every cell inside a square of ``side_cells`` cells
    centred on ``centre`` (all in feature-grid units)."""
    c = np.asarray(centres_xy, dtype=float).reshape(-1, 2)
    cx, cy = float(centre[0]), float(centre[1])
    half = 0.5 * float(side_cells) * float(stride)
    return ((c[:, 0] >= cx - half) & (c[:, 0] <= cx + half)
            & (c[:, 1] >= cy - half) & (c[:, 1] <= cy + half))


def apply_logit_mask(logits, index, fraction=1.0):
    """Apply the paper's lesion to pre-softmax logits.

    A fully removed entry gets ``-inf``; a fractionally removed entry gets
    ``log(1 - fraction)``.  ``fraction == 0`` returns an unchanged copy, which
    is the zero-dose identity check used throughout the paper.
    """
    out = np.array(logits, dtype=float, copy=True)
    index = np.asarray(index, dtype=int).ravel()
    frac = np.broadcast_to(np.asarray(fraction, dtype=float), index.shape).copy()
    for cell, f in zip(index, frac):
        if f <= 0.0:
            continue
        if f >= 1.0:
            out[cell] = -np.inf
        else:
            out[cell] = out[cell] + np.log1p(-f)
    return out
