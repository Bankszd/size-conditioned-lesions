"""The bootstrap must treat a k-times drawn image exactly like k copies."""

import numpy as np

from lesions.bootstrap import draw_matrix, image_weights, percentile_interval, size_gap


def _toy(n_images=40, seed=3):
    rng = np.random.default_rng(seed)
    bins = np.array(["small", "medium", "large"])[rng.integers(0, 3, 200)]
    images = rng.integers(0, n_images, 200)
    loss = rng.normal(size=200)
    return loss, bins, images, n_images


def test_duplicated_image_equals_physically_duplicated_rows():
    loss, bins, images, n_images = _toy()
    for draw in draw_matrix(n_images, 5, seed=11):
        counts = image_weights(draw, n_images)
        weighted = size_gap(loss, bins, images, counts)

        order = np.concatenate([np.flatnonzero(images == i).repeat(int(c))
                                for i, c in enumerate(counts) if c > 0])
        duplicated = size_gap(loss[order], bins[order], images[order],
                              np.ones(order.size))
        assert abs(weighted - duplicated) < 1e-12


def test_draw_matrix_is_reproducible_and_full_size():
    a = draw_matrix(4000, 2000)
    b = draw_matrix(4000, 2000)
    assert a.shape == (2000, 4000)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 4000


def test_draw_matrix_actually_repeats_images():
    draw = draw_matrix(4000, 1)[0]
    assert len(set(draw.tolist())) < 4000


def test_size_gap_recovers_a_known_offset():
    bins = np.array(["small"] * 10 + ["large"] * 10)
    images = np.arange(20)
    loss = np.array([1.0] * 10 + [0.0] * 10)
    assert abs(size_gap(loss, bins, images, np.ones(20)) - 1.0) < 1e-12


def test_percentile_interval_is_symmetric_on_a_symmetric_sample():
    x = np.linspace(-1.0, 1.0, 10001)
    lo, hi = percentile_interval(x)
    assert abs(lo + hi) < 1e-9


def test_percentile_interval_drops_non_finite_draws():
    x = np.array([np.nan, np.inf, 1.0, 2.0, 3.0])
    lo, hi = percentile_interval(x)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo >= 1.0 and hi <= 3.0
