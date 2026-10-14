"""Fast path: one-time CSV -> tiny int16 binary, then detection in well under a second.

convert(): pyarrow multithreaded CSV stream, mean-decimate 50:1 (50 kHz -> 1 kHz),
           subtract a slow block-median baseline, store int16 [n, 2] (adc1, adc2).
detect():  memory-map the .bin, skip quiet 100 ms blocks, find_peaks only in active
           blocks, group peaks < 1 s apart into pieces, return features.
"""
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

BLOCK = 50          # raw samples per output sample
FS = 1000.0
NO_DYE = 69.0


def _baseline(x, fs=FS, win_s=2.0):
    b = int(win_s * fs); n = len(x) // b
    if n < 2:
        return np.full(len(x), np.median(x))
    med = np.median(x[:n * b].reshape(n, b), axis=1)
    return np.interp(np.arange(len(x)), (np.arange(n) + 0.5) * b, med)


def convert(csv, out_dir):
    import pyarrow as pa          # only needed for CSV conversion, not for detection
    import pyarrow.csv as pc
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv = Path(csv)
    ro = pc.ReadOptions(block_size=64 << 20)
    co = pc.ConvertOptions(column_types={"adc1": pa.int16(), "adc2": pa.int16()})
    parts, carry = [], np.empty((0, 2), np.int16)
    for b in pc.open_csv(csv, read_options=ro, convert_options=co):
        a = np.column_stack([b.column(0).to_numpy(), b.column(1).to_numpy()])
        a = np.concatenate([carry, a]) if len(carry) else a
        n = len(a) // BLOCK * BLOCK
        carry = a[n:]
        parts.append(a[:n].reshape(-1, BLOCK, 2).mean(1, dtype=np.float32))
    x = np.concatenate(parts)
    y = np.column_stack([x[:, i] - _baseline(x[:, i]) for i in range(2)])
    np.clip(np.rint(y), -32768, 32767).astype(np.int16).tofile(out_dir / (csv.stem + ".bin"))
    (out_dir / (csv.stem + ".json")).write_text(json.dumps({"fs": FS, "n": len(x), "block": BLOCK}))


def _sigma(col, blk, stride=8):
    """Quietest-10% block sigma, estimated from every `stride`-th 100 ms block."""
    n = len(col) // blk
    v = col[:n * blk].reshape(n, blk)[::stride].astype(np.float32)
    return float(np.sqrt(np.mean(np.sort(v.std(1))[:max(1, len(v) // 10)] ** 2)))


def detect(bin_path, k=30.0, blk=100, gap_s=1.0, no_dye=NO_DYE):
    a = np.memmap(bin_path, dtype=np.int16, mode="r").reshape(-1, 2)
    x, y = a[:, 1], a[:, 0]                           # adc2, adc1 (strided views)
    n = len(x) // blk
    sigma, thr = _sigma(x, blk), 0.0
    thr = k * sigma
    mx = x[:n * blk].reshape(n, blk).max(1)           # int16, no float copy
    act = np.flatnonzero(mx > thr)                    # skip quiet blocks
    peaks = []
    for run in np.split(act, np.flatnonzero(np.diff(act) > 1) + 1) if len(act) else []:
        lo, hi = max(0, run[0] * blk - 5), min(len(x), (run[-1] + 1) * blk + 5)
        pk, _ = find_peaks(x[lo:hi].astype(np.float32), height=thr, distance=5)
        peaks.append(pk + lo)
    p = np.concatenate(peaks) if peaks else np.empty(0, int)
    if len(p) == 0:
        return {"sigma": sigma, "peaks": 0, "pieces": 0, "dyed": 0, "solid_no_dye": 0,
                "water": 0, "sum_height": 0.0, "dyed_height": 0.0,
                "peak_t": p / FS, "peak_h": np.empty(0)}
    groups = np.split(p, np.flatnonzero(np.diff(p) > gap_s * FS) + 1)
    h = x[p].astype(np.float32)
    mxg = np.array([x[g].max() for g in groups], dtype=np.float32)
    thr1 = k * _sigma(y, blk)
    dim = [i for i in range(len(groups)) if mxg[i] <= no_dye]
    solid = sum(int(y[max(0, groups[i][0] - 20):groups[i][-1] + 21].max() > thr1) for i in dim)
    return {"sigma": sigma, "peaks": len(p), "pieces": len(groups),
            "dyed": int((mxg > no_dye).sum()), "solid_no_dye": solid,
            "water": len(dim) - solid, "sum_height": float(h.sum()),
            "dyed_height": float(mxg[mxg > no_dye].sum()),
            "peak_t": p / FS, "peak_h": h}
