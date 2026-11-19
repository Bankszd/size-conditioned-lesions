"""Zero-dose identity for the lesion arithmetic the hooks apply.

The decoder hooks themselves need a checkpoint and therefore live in
``experiments/``.  They implement the paper's rule by adding ``-inf`` (or
``log(1 - f)``) to the selected pre-softmax entries and then renormalising with
the ordinary softmax.  These tests pin the arithmetic that the hooks rely on;
they run without torch so that ``make test`` works in the CPU-only
environment.
"""

import numpy as np
import pytest

from lesions.dose import apply_logit_mask


def _softmax(x):
    # A fully masked row has no defined distribution; the softmax of an all
    # -inf row is taken to be the zero vector rather than nan.
    top = np.max(x)
    if not np.isfinite(top):
        return np.zeros_like(x)
    e = np.exp(x - top)
    return e / e.sum()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_zero_dose_is_bit_identical(seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=32)
    idx = rng.permutation(32)[:6]
    out = apply_logit_mask(logits, idx, np.zeros(idx.size))
    assert np.array_equal(out, logits)
    assert np.array_equal(_softmax(out), _softmax(logits))


def test_a_non_trivial_lesion_changes_the_output():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=32)
    idx = np.arange(8)
    out = apply_logit_mask(logits, idx, np.ones(idx.size))
    assert not np.allclose(_softmax(out), _softmax(logits))
    assert np.all(_softmax(out)[idx] == 0.0)


def test_all_cells_removed_is_degenerate_but_finite_elsewhere():
    logits = np.array([0.1, 0.2, 0.3])
    out = apply_logit_mask(logits, [0, 1, 2], [1.0, 1.0, 1.0])
    probs = _softmax(out)
    assert np.all(probs == 0.0)


def test_half_dose_on_every_cell_leaves_the_distribution_unchanged():
    # scaling all logits by the same additive constant must not change softmax
    logits = np.array([0.5, -0.5, 1.5, 0.0])
    out = apply_logit_mask(logits, [0, 1, 2, 3], [0.5] * 4)
    assert np.allclose(_softmax(out), _softmax(logits))
