"""Drop-in faster versions of the group's Part A / Part B hot spots (same results, to be verified).

Where the time went on the 4.7 h recording (17 M samples, 1 kHz):
  Part A per channel: medfilt 1.0 s | savgol 2.6 s | ALS (sparse spsolve) 3.6 s
  Part B: find_peaks(prominence=...) over the WHOLE recording, adc2 3.7 s + adc1 2.4 s

Changes:
  * median(3)            -> vectorised min/max (same values, zero padding at the edges like medfilt)
  * Savitzky-Golay(7,3)  -> scipy's savgol_filter (unchanged)
  * ALS                  -> banded Cholesky (scipy.linalg.solveh_banded) instead of sparse spsolve
  * both channels in two threads (Part A and the two find_peaks calls in Part B)

Tried and dropped: running find_peaks(prominence) only near activity. It is only exact with windows so wide
(>= 10 s) that it saves nothing; with 1 s windows it missed 82 peaks over 28 recordings.
"""
import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import find_peaks, savgol_filter


def median3(x):
    x = np.asarray(x, np.float64)
    p = np.concatenate([[0.0], x, [0.0]])                            # medfilt zero-pads
    a, b, c = p[:-2], p[1:-1], p[2:]
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def als_banded(y, lam, p, n_iter):
    n = len(y)
    d0 = np.full(n, 6.0); d0[[0, -1]] = 1.0; d0[[1, -2]] = 5.0
    d1 = np.full(n - 1, -4.0); d1[[0, -1]] = -2.0
    d2 = np.ones(n - 2)
    w = np.ones(n); z = y
    for _ in range(n_iter):
        ab = np.zeros((3, n))
        ab[2] = w + lam * d0
        ab[1, 1:] = lam * d1
        ab[0, 2:] = lam * d2
        z = solveh_banded(ab, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def baseline_fast(y, params):
    b = params["als_block"]
    n = len(y) // b * b
    c = y[:n].reshape(-1, b).mean(1, dtype=np.float64)
    z = als_banded(c, params["als_lam"], params["als_p"], params["als_iter"])
    return np.interp(np.arange(len(y)), (np.arange(len(c)) + 0.5) * b, z)


def clean_channel_fast(x, params):
    s = savgol_filter(median3(x), params["sg_window"], params["sg_order"])
    return s - baseline_fast(s, params)


def preprocess_fast(mean_1khz, fs=1000.0, params=None):
    """Drop-in for preprocessing.preprocess: both channels cleaned in two threads."""
    from concurrent.futures import ThreadPoolExecutor
    from core import preprocessing as P          # the group's own module (sigma, adc1_response)
    params = params or P.PARAMS
    with ThreadPoolExecutor(2) as ex:
        a1, a2 = ex.map(lambda ch: clean_channel_fast(mean_1khz[:, ch], params), (0, 1))
    a1, a2 = a1.astype(np.float32), a2.astype(np.float32)        # same dtype as the group's output
    r = P.adc1_response(a1, a2, fs)
    return {"fs": fs, "adc1": a1, "adc2": a2,
            "sigma_adc1": P.quiet_sigma(a1, fs), "sigma_adc2": P.quiet_sigma(a2, fs),
            "adc1_response": r, "adc1_usable": bool(r >= P.ADC1_MIN_RESPONSE)}


def detect_both(a1, s1, a2, s2, fs, k, detect_peaks):
    """Peak detection on adc1 and adc2 in two threads (same function, same result)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        f1 = ex.submit(detect_peaks, a1, s1, fs, k); f2 = ex.submit(detect_peaks, a2, s2, fs, k)
        return f1.result(), f2.result()
