"""Whole-image percentile bootstrap.

Every interval in the paper resamples images, never anchors: all anchors of a
drawn image move together.  A drawn image that appears k times enters the
statistic k times, which is exactly what physically duplicating its rows would
do (``tests/test_bootstrap.py`` checks this).
"""

import numpy as np

SEED = 20271001


def draw_matrix(n_images, n_draws, seed=SEED):
    """(n_draws, n_images) matrix of image positions, drawn with replacement."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, int(n_images), size=(int(n_draws), int(n_images)))


def image_weights(draw, n_images):
    """Multiplicity of every image in one draw."""
    return np.bincount(np.asarray(draw, dtype=int),
                       minlength=int(n_images)).astype(float)


def _mean(values, weights):
    w = float(np.sum(weights))
    if w <= 0:
        return np.nan
    return float(np.sum(np.asarray(values, float) * weights) / w)


def size_gap(loss, bins, image_index, weights):
    """E[loss | small] - E[loss | large] under image weights."""
    loss = np.asarray(loss, float)
    bins = np.asarray(bins)
    image_index = np.asarray(image_index, int)
    w = np.asarray(weights, float)[image_index]
    small = bins == "small"
    large = bins == "large"
    return _mean(loss[small], w[small]) - _mean(loss[large], w[large])


def percentile_interval(values, level=95.0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    lo = (100.0 - level) / 2.0
    return [float(np.percentile(v, lo)), float(np.percentile(v, 100.0 - lo))]
