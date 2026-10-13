"""Preprocessing: peak-preserving decimation, drift baseline, noise floor.

Raw CSV (adc1, adc2 at ~50 kHz) -> compact .npz with per-block min/max/mean
for both channels, so 5 ms spikes survive decimation (mean-pooling alone
keeps only ~36% of peak height at 100 Hz).
"""
from pathlib import Path
import numpy as np
import pandas as pd

CFG = {
    "raw_fs": 50_000,        # Hz (nominal; override with measured duration)
    "block": 50,             # raw samples per output sample (-> ~1 kHz)
    "chunk_rows": 5_000_000,
    "channels": ("adc1", "adc2"),
    "baseline_window_s": 60.0,
    "k_mad": 5.0,            # noise floor = baseline + k * MAD
}


def decimate_csv(path, duration_s=None, cfg=CFG):
    """Stream a CSV in chunks; return dict of per-block mean/min/max (float32)."""
    block = cfg["block"]
    out = {f"{c}_{s}": [] for c in cfg["channels"] for s in ("mean", "min", "max")}
    carry = None
    n_raw = 0
    for chunk in pd.read_csv(path, usecols=list(cfg["channels"]), dtype=np.float32,
                             chunksize=cfg["chunk_rows"]):
        chunk = chunk.dropna()
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        n = len(chunk) // block * block
        carry = chunk.iloc[n:]
        n_raw += n
        for c in cfg["channels"]:
            x = chunk[c].to_numpy()[:n].reshape(-1, block)
            out[f"{c}_mean"].append(x.mean(1))
            out[f"{c}_min"].append(x.min(1))
            out[f"{c}_max"].append(x.max(1))
    res = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    fs_raw = n_raw / duration_s if duration_s else cfg["raw_fs"]
    res["fs"] = np.float64(fs_raw / block)
    return res


def rolling_median_baseline(x, fs, window_s):
    """Slow-drift baseline; window long vs. spike clusters."""
    w = max(3, int(window_s * fs) | 1)
    return pd.Series(x).rolling(w, center=True, min_periods=1).median().to_numpy()


def noise_floor(x, baseline, k=CFG["k_mad"]):
    """Limit of detection from a blank (zero/water) segment: k * MAD."""
    r = x - baseline
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    return k * mad, mad


def preprocess(path, out_npz, duration_s=None, cfg=CFG):
    d = decimate_csv(path, duration_s, cfg)
    fs = float(d["fs"])
    for c in cfg["channels"]:
        d[f"{c}_baseline"] = rolling_median_baseline(
            d[f"{c}_mean"], fs, cfg["baseline_window_s"]).astype(np.float32)
    np.savez_compressed(out_npz, **d)
    return d
