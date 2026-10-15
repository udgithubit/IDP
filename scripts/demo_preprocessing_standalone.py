"""Self-contained demo of the preprocessing stage (noise reduction, baseline, threshold).

    python demo_preprocessing_standalone.py "PATH\\TO\\recording.csv" [demo_preprocessing.png]

Needs only: pip install numpy pandas scipy matplotlib
Steps: 50 kHz -> 1 kHz (mean of 50 samples) | median(3) + Savitzky-Golay(7, 3) | ALS baseline | sigma, threshold.
"""
import sys, time
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import medfilt, savgol_filter, find_peaks

# ---- parameters (edit here) ------------------------------------------------
RAW_FS, BLOCK = 50_000, 50          # 50 kHz -> 1 kHz
MED, SG_WIN, SG_ORDER = 3, 7, 3     # noise reduction
ALS_LAM, ALS_P, ALS_DS = 1e5, 0.01, 100   # baseline (fitted on 10 Hz block means)
K = 30                              # threshold = K * sigma
# -----------------------------------------------------------------------------

def decimate(csv):
    out, carry = [], None
    for ch in pd.read_csv(csv, usecols=["adc1", "adc2"], dtype=np.float32, chunksize=5_000_000):
        ch = ch.dropna()
        if carry is not None:
            ch = pd.concat([carry, ch], ignore_index=True)
        n = len(ch) // BLOCK * BLOCK
        carry = ch.iloc[n:]
        out.append(ch.iloc[:n].to_numpy().reshape(-1, BLOCK, 2).mean(1))
    return np.concatenate(out).astype(np.float64)          # [n, 2] at 1 kHz

def als(y, lam=ALS_LAM, p=ALS_P, n_iter=10):
    n = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    H = lam * (D.T @ D); w = np.ones(n)
    for _ in range(n_iter):
        z = spsolve((sparse.diags(w) + H).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z

def baseline(y):
    n = len(y) // ALS_DS * ALS_DS
    c = y[:n].reshape(-1, ALS_DS).mean(1)
    return np.interp(np.arange(len(y)), (np.arange(len(c)) + .5) * ALS_DS, als(c))

def quiet_sigma(x, fs=1000.0, block_s=0.2, frac=0.10):
    b = int(block_s * fs); n = len(x) // b
    s = np.sort(x[:n * b].reshape(n, b).std(1))
    return float(np.sqrt(np.mean(s[:max(1, int(frac * n))] ** 2)))

if __name__ == "__main__":
    csv = sys.argv[1]; png = sys.argv[2] if len(sys.argv) > 2 else "demo_preprocessing.png"
    t0 = time.perf_counter(); x = decimate(csv); t_dec = time.perf_counter() - t0
    fs = RAW_FS / BLOCK; t = np.arange(len(x)) / fs
    t1 = time.perf_counter(); res = {}
    for i, name in enumerate(("adc1", "adc2")):
        raw = x[:, i]
        den = savgol_filter(medfilt(raw, MED), SG_WIN, SG_ORDER)
        base = baseline(den); cor = den - base; s = quiet_sigma(cor, fs)
        res[name] = (raw, den, base, cor, s)
    t_clean = time.perf_counter() - t1
    print(f"{len(x):,} samples at {fs:.0f} Hz ({len(x)/fs/60:.1f} min)")
    print(f"CSV -> 1 kHz: {t_dec:.1f} s | noise reduction + baseline + sigma (both channels): {t_clean:.2f} s")
    for n, (_, _, _, _, s) in res.items():
        print(f"{n}: sigma = {s:.2f} counts, threshold = {K} x sigma = {K * s:.1f} counts")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    raw, den, base, cor, s = res["adc2"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    ax[0, 0].plot(t, raw, lw=.5, label="1 kHz signal"); ax[0, 0].plot(t, base, "r", lw=1.2, label="ALS baseline")
    ax[0, 0].set(title="1. Slow drift is fitted under the peaks", xlabel="s", ylabel="adc2 (counts)"); ax[0, 0].legend()
    ax[0, 1].plot(t, cor, lw=.5, color="k"); ax[0, 1].axhline(K * s, color="r", ls="--", label=f"threshold {K*s:.0f} ({K} sigma)")
    ax[0, 1].set(title="2. Baseline-corrected signal with threshold", xlabel="s", ylabel="counts"); ax[0, 1].legend()
    k = int(np.argmax(cor)); a, b = max(0, k - 400), min(len(cor), k + 700)
    ax[1, 0].plot(t[a:b], raw[a:b] - base[a:b], ".-", ms=3, lw=.8, label="before"); ax[1, 0].plot(t[a:b], cor[a:b], lw=1.5, label="after median + Savitzky-Golay")
    ax[1, 0].set(title="3. Noise reduction keeps the peak shape (zoom on the tallest peak)", xlabel="s"); ax[1, 0].legend()
    ks = [5, 8, 10, 15, 20, 30, 50]
    ax[1, 1].plot(ks, [len(find_peaks(cor, height=kk * s, distance=5)[0]) for kk in ks], "o-")
    ax[1, 1].set(title="4. Peaks found in the whole recording vs threshold k x sigma", xlabel="k", ylabel="peaks")
    plt.tight_layout(); plt.savefig(png, dpi=110); print("saved", png)
