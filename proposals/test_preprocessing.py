"""Tests for the cleaning stage (core/preprocessing.py). Run from the group repo root:  python -m pytest -q proposals/test_preprocessing.py
They need numpy and scipy only (no recordings)."""
import numpy as np
import pytest
from scipy.signal import medfilt

from core import preprocessing as P


def test_median_filter_matches_scipy():
    x = np.random.default_rng(0).normal(0, 1, 1000)
    for size in (3, 5):
        assert np.array_equal(P.median_filter(x, size), medfilt(x, size))


@pytest.mark.parametrize("n", [4, 5, 10, 100, 1000])
def test_banded_als_matches_sparse_solver(n):
    y = np.random.default_rng(n).normal(0, 1, n) + np.linspace(0, 5, n)
    assert np.allclose(P.als(y, 1e5, 0.01, 10), P._als_sparse(y, 1e5, 0.01, 10), atol=1e-6)


def test_als_short_inputs_do_not_crash():
    assert len(P.als(np.array([1.0, 2.0]), 1e5, 0.01, 10)) == 2
    assert len(P.als(np.array([1.0, 2.0, 3.0]), 1e5, 0.01, 10)) == 3


def test_baseline_removes_a_slow_drift_and_keeps_a_peak():
    t = np.arange(60_000) / 1000.0
    x = 600 + 0.5 * t + np.random.default_rng(1).normal(0, 1, t.size)
    x[29_998:30_003] += np.array([200, 600, 900, 600, 200])   # a 5 ms wide peak, like the real ones
    clean = P.clean_channel(x)
    assert abs(np.median(clean)) < 5                   # drift gone, baseline near 0
    assert clean[29_990:30_010].max() > 600            # the peak survives (the filters trim some height)


def test_single_sample_spike_is_removed_by_the_median_filter():
    x = np.full(60_000, 600.0)
    x[30_000] += 800                                   # a one-sample spike (1 ms), not a real peak
    assert P.clean_channel(x)[29_990:30_010].max() < 20


def test_quiet_sigma_of_white_noise_is_close_to_its_std():
    x = np.random.default_rng(2).normal(0, 2.0, 100_000)
    assert 1.5 < P.quiet_sigma(x) < 2.1                # the quietest blocks are a bit below the true std


def test_quiet_sigma_ignores_idle_blocks():
    rng = np.random.default_rng(3)
    x = np.concatenate([np.zeros(50_000), rng.normal(0, 2.0, 50_000)])   # half the recording is idle
    assert P.quiet_sigma(x) > 1.0


def test_preprocess_output_format():
    m = np.random.default_rng(4).normal(500, 3, (5_000, 2)).astype(np.float32)
    out = P.preprocess(m)
    assert out["adc1"].dtype == np.float32 and out["adc2"].dtype == np.float32
    assert len(out["adc1"]) == len(out["adc2"]) == 5_000
    assert out["sigma_adc1"] > 0 and out["sigma_adc2"] > 0
