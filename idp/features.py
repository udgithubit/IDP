"""Peak detection + per-file features on preprocessed .npz (adc2 max envelope)."""
import numpy as np
from scipy.signal import find_peaks
from .preprocessing import noise_floor


def file_features(npz, ch="adc2", k=5.0, min_gap_s=2.0):
    d = np.load(npz)
    fs = float(d["fs"])
    x = d[f"{ch}_max"] - d[f"{ch}_baseline"]
    lod, mad = noise_floor(d[f"{ch}_max"], d[f"{ch}_baseline"], k)
    lod = max(lod, 15.0)
    p, _ = find_peaks(x, height=lod, distance=max(1, int(0.05 * fs)))
    h = x[p]
    tp = p / fs
    gaps = np.diff(tp)
    n_events = (1 + int((gaps > min_gap_s).sum())) if len(p) else 0
    return {
        "duration_s": len(x) / fs, "mad": mad, "lod": lod,
        "n_peaks": len(p), "n_events": n_events,
        "sum_height": float(h.sum()), "mean_height": float(h.mean()) if len(h) else 0.0,
        "p90_height": float(np.percentile(h, 90)) if len(h) else 0.0,
        "area": float(np.clip(x - lod, 0, None).sum() / fs),
    }
