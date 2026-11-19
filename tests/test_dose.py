"""Unit tests for the lesion arithmetic in ``lesions/dose.py``."""

import numpy as np

from lesions.dose import (
    apply_logit_mask,
    box_cell_mask,
    cell_fraction_select,
    equal_mass_select,
    fixed_area_mask,
    fixed_count_select,
)


def test_fixed_count_returns_k_distinct_cells():
    rng = np.random.default_rng(0)
    idx = fixed_count_select(20, 4, rng)
    assert idx.size == 4
    assert len(set(idx.tolist())) == 4
    assert idx.max() < 20


def test_fixed_count_is_saturated_by_the_box():
    rng = np.random.default_rng(0)
    idx = fixed_count_select(2, 4, rng)
    assert sorted(idx.tolist()) == [0, 1]


def test_fixed_count_handles_empty_box():
    rng = np.random.default_rng(0)
    assert fixed_count_select(0, 4, rng).size == 0


def test_equal_mass_hits_the_target_exactly():
    masses = np.full(60, 0.004)
    idx, frac = equal_mass_select(masses, 0.02)
    removed = float((masses[idx] * frac).sum())
    assert abs(removed - 0.02) <= 1e-4
    assert idx.size == 5


def test_equal_mass_uses_a_partial_cell_when_needed():
    masses = np.array([0.01, 0.01, 0.01, 0.01])
    idx, frac = equal_mass_select(masses, 0.015)
    removed = float((masses[idx] * frac).sum())
    assert abs(removed - 0.015) <= 1e-4
    assert frac[-1] < 1.0


def test_equal_mass_deletes_everything_when_the_box_is_light():
    masses = np.array([0.001, 0.001, 0.001])
    idx, frac = equal_mass_select(masses, 0.5)
    assert idx.size == 3
    assert np.allclose(frac, 1.0)


def test_equal_mass_on_an_empty_box():
    idx, frac = equal_mass_select(np.zeros(0), 0.1)
    assert idx.size == 0 and frac.size == 0


def test_cell_fraction_even_and_odd():
    idx, frac = cell_fraction_select(8, 0.5)
    assert idx.size == 4 and np.allclose(frac, 1.0)

    idx, frac = cell_fraction_select(9, 0.5)
    assert idx.size == 5
    assert np.allclose(frac[:-1], 1.0)
    assert abs(float(frac[-1]) - 0.5) < 1e-12


def test_cell_fraction_zero_cells():
    idx, frac = cell_fraction_select(0, 0.5)
    assert idx.size == 0 and frac.size == 0


def test_box_cell_mask_uses_the_cell_centre():
    centres = np.array([[8, 8], [24, 8], [40, 8], [8, 40]], float)
    mask = box_cell_mask(centres, [0, 0, 32, 32])
    assert mask.tolist() == [True, True, False, False]
    mask = box_cell_mask(centres, [0, 0, 48, 48])
    assert mask.tolist() == [True, True, True, True]


def test_fixed_area_is_centred_and_square():
    stride = 16
    xs = np.arange(0, 20, 1) * stride + stride / 2.0
    centres = np.array([[x, y] for y in xs for x in xs], float)
    mask = fixed_area_mask(centres, (160.0, 160.0), side_cells=2, stride=stride)
    # a 2x2-cell square of side 32 px centred on (160, 160)
    selected = centres[mask]
    assert selected.shape[0] == 4
    assert np.all(np.abs(selected - 160.0) <= stride)


def test_zero_dose_is_the_identity():
    logits = np.array([0.3, -1.2, 2.0, 0.0, 5.5])
    base = np.exp(logits) / np.exp(logits).sum()
    out = apply_logit_mask(logits, index=[1, 3], fraction=[0.0, 0.0])
    got = np.exp(out) / np.exp(out).sum()
    assert np.array_equal(base, got)
    assert np.array_equal(out, logits)


def test_full_dose_removes_the_mass():
    logits = np.array([0.3, -1.2, 2.0, 0.0, 5.5])
    out = apply_logit_mask(logits, index=[4], fraction=[1.0])
    probs = np.exp(out) / np.exp(out).sum()
    assert probs[4] == 0.0
    assert abs(probs.sum() - 1.0) < 1e-12


def test_partial_dose_matches_log_one_minus_f():
    logits = np.array([1.0, 1.0])
    out = apply_logit_mask(logits, index=[0], fraction=[0.25])
    assert abs(out[0] - (1.0 + np.log(0.75))) < 1e-12
