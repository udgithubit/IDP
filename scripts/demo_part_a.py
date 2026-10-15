"""Demo of the preprocessing stage (noise reduction, baseline, threshold) on one recording.
Usage: python scripts/demo_part_a.py CSV [OUT_PNG]    (e.g. the Big pieces CSV, 151 MB)
Prints timings and sigma/threshold; saves a 4-panel figure."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.signal import medfilt, savgol_filter, find_peaks
from idp.preprocessing import decimate_csv
from idp import method as M

csv = sys.argv[1]; png = sys.argv[2] if len(sys.argv) > 2 else "demo_part_a.png"
t0 = time.perf_counter(); d = decimate_csv(csv); t_dec = time.perf_counter() - t0
fs = 1000.0; raw = d["adc2_mean"].astype(np.float64); t = np.arange(len(raw)) / fs
t1 = time.perf_counter()
den = savgol_filter(medfilt(raw, M.P["med"]), M.P["sg_win"], M.P["sg_order"])
base = M.baseline(den, fs); cor = den - base
sigma = M.quiet_sigma(cor, fs); thr = M.P["k"] * sigma
t_clean = time.perf_counter() - t1
print(f"CSV -> 1 kHz (50:1): {t_dec:.1f} s | denoise + ALS baseline + sigma: {t_clean:.2f} s")
print(f"sigma = {sigma:.2f} counts, threshold = {M.P['k']:.0f} x sigma = {thr:.1f} counts")

def n_peaks(seg, k):
    return len(find_peaks(seg, height=k * sigma, distance=5)[0])
zero, water = (t >= 225) & (t <= 270), (t >= 285) & (t <= 335)     # blank segments of the Big pieces file
ks = [5, 8, 10, 15, 20, 30, 50]

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0, 0].plot(t, raw, lw=.5, label="1 kHz signal"); ax[0, 0].plot(t, base, "r", lw=1.2, label="ALS baseline (of the denoised signal)")
ax[0, 0].set(title="1. Slow drift is fitted under the peaks", xlabel="s", ylabel="adc2 (counts)"); ax[0, 0].legend()
ax[0, 1].plot(t, cor, lw=.5, color="k"); ax[0, 1].axhline(thr, color="r", ls="--", label=f"threshold {thr:.0f} (30 sigma)")
ax[0, 1].set(title="2. Baseline-corrected: flat at 0, ready for detection", xlabel="s", ylabel="counts"); ax[0, 1].legend()
a, b = 122.3, 123.4; m = (t >= a) & (t <= b)
ax[1, 0].plot(t[m], raw[m] - np.median(raw), ".-", ms=3, lw=.8, label="before (1 kHz mean)"); ax[1, 0].plot(t[m], den[m] - np.median(raw), "-", lw=1.5, label="after median + Savitzky-Golay")
ax[1, 0].set(title="3. Noise reduction keeps the peak shape (zoom on one piece)", xlabel="s"); ax[1, 0].legend()
ax[1, 1].plot(ks, [n_peaks(cor[water], k) for k in ks], "o-", label="water only (no pieces: ideally 0)")
ax[1, 1].plot(ks, [n_peaks(cor[zero], k) for k in ks], "s-", label="zero-dye segment (real tissue, not dyed)")
ax[1, 1].set(title="4. Peaks found in blank segments vs threshold k x sigma", xlabel="k", ylabel="peaks found"); ax[1, 1].legend()
plt.tight_layout(); plt.savefig(png, dpi=110); print("saved", png)
