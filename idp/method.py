"""Slide-deck method: clean signal (Part A) -> pieces/events (Part B).

A: decimate to 1 kHz -> median -> Savitzky-Golay -> ALS baseline; sigma = std of the
   quietest 10% of 200 ms blocks (per channel).
B: peaks > k*sigma; peaks < 1 s apart = one piece; piece brighter than the no-dye
   level (top 0.1% of water peaks) = dyed; dim pieces with an adc1 peak within
   +-20 ms of an adc2 peak = solid_no_dye, else water; amount = sum of dyed areas.
"""
import numpy as np
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

P = {"med": 3, "sg_win": 7, "sg_order": 3, "als_lam": 1e5, "als_p": 0.01,
     "als_ds": 100, "k": 30.0, "gap_s": 1.0, "pair_s": 0.020, "min_dist_s": 0.005}


def als(y, lam, p, n_iter=10):
    n = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    H = lam * (D.T @ D)
    w = np.ones(n)
    for _ in range(n_iter):
        z = spsolve((sparse.diags(w) + H).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def baseline(y, fs, cfg=P):
    ds = cfg["als_ds"]; n = len(y) // ds * ds
    c = y[:n].reshape(-1, ds).mean(1)
    z = als(c, cfg["als_lam"], cfg["als_p"])
    xs = (np.arange(len(c)) + 0.5) * ds
    return np.interp(np.arange(len(y)), xs, z)


def clean(x, fs, cfg=P):
    x = medfilt(x.astype(np.float64), cfg["med"])
    x = savgol_filter(x, cfg["sg_win"], cfg["sg_order"])
    return x - baseline(x, fs, cfg)


def quiet_sigma(x, fs, block_s=0.2, frac=0.10):
    b = int(block_s * fs); n = len(x) // b
    s = np.sort(x[:n * b].reshape(n, b).std(1))
    return float(np.sqrt(np.mean(s[:max(1, int(frac * n))] ** 2)))


def detect(x, sigma, fs, cfg=P):
    thr = cfg["k"] * sigma
    p, _ = find_peaks(x, height=thr, distance=max(1, int(cfg["min_dist_s"] * fs)))
    return p, thr


def run_signal(adc1_mean, adc2_mean, fs=1000.0, cfg=P):
    a1, a2 = clean(adc1_mean, fs, cfg), clean(adc2_mean, fs, cfg)
    s1, s2 = quiet_sigma(a1, fs), quiet_sigma(a2, fs)
    return {"a1": a1, "a2": a2, "sigma": {"adc1": s1, "adc2": s2}, "fs": fs}


def pieces(sig, no_dye_level, cfg=P):
    fs = sig["fs"]; a1, a2 = sig["a1"], sig["a2"]
    p2, thr2 = detect(a2, sig["sigma"]["adc2"], fs, cfg)
    p1, _ = detect(a1, sig["sigma"]["adc1"], fs, cfg)
    if len(p2) == 0:
        return [], {"dyed": 0, "solid_no_dye": 0, "water": 0}
    split = np.where(np.diff(p2) > cfg["gap_s"] * fs)[0] + 1
    groups = np.split(p2, split)
    out = []
    pw = int(cfg["pair_s"] * fs)
    for g in groups:
        lo, hi = g[0], g[-1]
        # extent: samples above threshold around first/last peak
        s, e = lo, hi
        while s > 0 and a2[s - 1] > thr2: s -= 1
        while e < len(a2) - 1 and a2[e + 1] > thr2: e += 1
        seg = a2[s:e + 1]; area = float(seg[seg > thr2].sum())
        mx = float(a2[g].max())
        if mx > no_dye_level: label = "dyed"
        else:
            hit = np.any((p1[None, :] >= (g[:, None] - pw)) & (p1[None, :] <= (g[:, None] + pw))) if len(p1) else False
            label = "solid_no_dye" if hit else "water"
        out.append({"start_s": s / fs, "n_peaks": len(g), "max_height": mx, "area": area, "label": label})
    summ = {l: sum(o["label"] == l for o in out) for l in ("dyed", "solid_no_dye", "water")}
    return out, summ
