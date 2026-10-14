"""Preprocessing output (Part A) that the detection / clustering stage consumes."""
from pathlib import Path
import numpy as np


def load_clean(path):
    """Return dict: fs (Hz), adc1, adc2 (baseline-corrected, float32, counts), sigma {adc1, adc2},
    thr {adc1, adc2} (= k * sigma), k, source, t (seconds, computed)."""
    d = np.load(path, allow_pickle=False)
    sc = np.float32(d["scale"])      # stored as int16 in 0.1-count steps
    out = {"fs": float(d["fs"]), "adc1": d["adc1_i16"] * sc, "adc2": d["adc2_i16"] * sc,
           "sigma": {"adc1": float(d["sigma_adc1"]), "adc2": float(d["sigma_adc2"])},
           "k": float(d["k"]), "source": str(d["source"])}
    out["thr"] = {c: out["k"] * out["sigma"][c] for c in ("adc1", "adc2")}
    out["t"] = np.arange(len(out["adc1"])) / out["fs"]
    return out
