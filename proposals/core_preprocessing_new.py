"""Part A: clean the 1 kHz signal from ``fast_io.summarize`` before peak detection.

    summ  = fast_io.summarize(idx)          # 1 kHz block means, (n, 2) adc1 / adc2
    clean = preprocess(summ["mean"])        # denoised, baseline = 0, sigma per channel

Steps per channel: median (3) -> Savitzky-Golay (7, order 3) -> ALS baseline subtracted,
then sigma = noise level of the quietest 10% of 200 ms blocks, ignoring idle blocks (no flow).

Contributions (see the README at the repository root):
- The processing itself is Unmona's preprocessing (udgithubit/IDP, idp/method.py), marked below.
- Input from fast_io and the output dict for detection.py: Sora.
- Dropping idle blocks before sigma (GitHub #3): Sora, marked inside quiet_sigma.
- adc1 response check (GitHub #4): Sora.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse
from scipy.linalg import solveh_banded
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.sparse.linalg import spsolve

FS_HZ = 1000.0

# ---- Idea: Unmona (udgithubit/IDP, idp/method.py: P, als, baseline, clean, quiet_sigma) ---------
# Median + Savitzky-Golay denoising, then an ALS baseline fitted on 10 Hz block means and
# interpolated back to 1 kHz (fitting ALS at 10 Hz is what makes it take about 1 s on 4.7 h).
# sigma: RMS of the std of the quietest 10% of 200 ms blocks.
# The values are Unmona's first guesses; the slide values may replace them.
PARAMS = {"median": 3, "sg_window": 7, "sg_order": 3,
          "als_lam": 1e5, "als_p": 0.01, "als_iter": 10, "als_block": 100}


def _als_sparse(y: np.ndarray, lam: float, p: float, n_iter: int) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens 2005), sparse solver (reference version)."""
    n = len(y)
    d = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    h = lam * (d.T @ d)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        z = spsolve((sparse.diags(w) + h).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


# -- Speed-up (Unmona, written with Claude; verified identical pieces and labels on 28-32 recordings) --
# The same ALS system is pentadiagonal and symmetric, so a banded Cholesky solve replaces the sparse
# solver (1.9 s -> 0.6 s on 4.7 h). The band layout below is for n >= 4; shorter inputs use the sparse
# version (n = 3) or get no baseline (n < 3, nothing to fit).
def als(y: np.ndarray, lam: float, p: float, n_iter: int) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens 2005)."""
    n = len(y)
    if n < 3:
        return np.asarray(y, np.float64)
    if n < 4:
        return _als_sparse(y, lam, p, n_iter)
    d0 = np.full(n, 6.0)
    d0[[0, -1]] = 1.0
    d0[[1, -2]] = 5.0
    d1 = np.full(n - 1, -4.0)
    d1[[0, -1]] = -2.0
    d2 = np.ones(n - 2)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        ab = np.zeros((3, n))                      # upper band storage of  W + lam * D'D
        ab[2] = w + lam * d0
        ab[1, 1:] = lam * d1
        ab[0, 2:] = lam * d2
        z = solveh_banded(ab, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def median_filter(x: np.ndarray, size: int) -> np.ndarray:
    """Same values as scipy.signal.medfilt (zero padding at the edges); size 3 is done with min/max."""
    if size != 3:
        return medfilt(np.asarray(x, np.float64), size)
    p = np.concatenate([[0.0], np.asarray(x, np.float64), [0.0]])
    a, b, c = p[:-2], p[1:-1], p[2:]
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
# -- end of speed-up ----------------------------------------------------------------------------------


def baseline(y: np.ndarray, params: dict = PARAMS) -> np.ndarray:
    """ALS on block means (10 Hz for 1 kHz input), interpolated back to every sample."""
    b = params["als_block"]
    n = len(y) // b * b
    c = y[:n].reshape(-1, b).mean(1)
    z = als(c, params["als_lam"], params["als_p"], params["als_iter"])
    return np.interp(np.arange(len(y)), (np.arange(len(c)) + 0.5) * b, z)


def clean_channel(x: np.ndarray, params: dict = PARAMS) -> np.ndarray:
    x = median_filter(x, params["median"])
    x = savgol_filter(x, params["sg_window"], params["sg_order"])
    return x - baseline(x, params)


def quiet_sigma(x: np.ndarray, fs: float = FS_HZ, block_s: float = 0.2, frac: float = 0.10,
                idle_cut: float = 0.5) -> float:
    b = int(block_s * fs)
    n = len(x) // b
    s = np.sort(x[: n * b].reshape(n, b).std(1))
    # -- Changed by Sora (GitHub #3): drop idle blocks first --------------------------------
    # Blocks with std below idle_cut x the median block std are periods with no flow. In
    # recordings with long idle periods they made sigma so small that the threshold fell below
    # the water wiggles and pieces merged ("good data hig middle low zero": 1 piece of 323 s
    # -> 48 pieces). 0.3-0.6 all fix that and keep the Big pieces results; 0.7 loses zero
    # pieces confirmed by adc1. 0.5 is the middle of that range (05_信号特性の実測/38_*.py).
    if idle_cut:
        s = s[s >= idle_cut * np.median(s)]
    # -- end of change ----------------------------------------------------------------------
    return float(np.sqrt(np.mean(s[: max(1, int(frac * len(s)))] ** 2)))
# ---- end of Unmona's idea ---------------------------------------------------------------------


ADC1_MIN_RESPONSE = 0.1


def adc1_response(adc1: np.ndarray, adc2: np.ndarray, fs: float = FS_HZ,
                  n_top: int = 50, window_s: float = 0.020) -> float:
    """Median of adc1 / adc2 at the n_top largest adc2 peaks (adc1 = its max within +-window_s).

    0.125-0.64 in the 2025-2026 recordings, 0.019-0.058 in the 2023 test signals, where adc1
    hardly responds (GitHub #4, 05_信号特性の実測/39_adc1_response.py). There the adc1 threshold
    is only about 2 counts and noise passes the two-channel check by chance, so below
    ADC1_MIN_RESPONSE adc1 is not used.
    """
    p, _ = find_peaks(adc2, distance=int(0.1 * fs))
    p = p[np.argsort(adc2[p])[-n_top:]]
    p = p[adc2[p] > 0]
    if len(p) == 0:
        return float("nan")
    w = int(window_s * fs)
    return float(np.median([adc1[max(0, i - w): i + w + 1].max() / adc2[i] for i in p]))


def preprocess(mean_1khz: np.ndarray, fs: float = FS_HZ, params: dict = PARAMS) -> dict:
    """(n, 2) 1 kHz block means -> cleaned adc1 / adc2 (float32), sigma per channel and
    whether adc1 responds well enough for the two-channel check."""
    # both channels in two threads (numpy and LAPACK release the GIL); the result is the same as in sequence
    with ThreadPoolExecutor(2) as ex:
        a1, a2 = ex.map(lambda ch: clean_channel(mean_1khz[:, ch], params), (0, 1))
    response = adc1_response(a1, a2, fs)
    return {
        "fs": fs,
        "adc1": a1.astype(np.float32),
        "adc2": a2.astype(np.float32),
        "sigma_adc1": quiet_sigma(a1, fs),         # sigma and response from the float64 signal, then cast
        "sigma_adc2": quiet_sigma(a2, fs),
        "adc1_response": response,
        "adc1_usable": bool(response >= ADC1_MIN_RESPONSE),   # NaN (no adc2 peaks) -> False
    }
