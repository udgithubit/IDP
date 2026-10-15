"""Compare two folders of pieces tables written by tools/run_pipeline.py --out (old core vs new core).

    python compare_outputs.py OLD_DIR NEW_DIR

Labels, piece counts, peak counts and start/end must be identical; max_height may differ by 0.1 and area by 2.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

a, b = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(p.name for p in a.glob("*.pieces.csv"))
bad = 0
for n in names:
    if not (b / n).exists():
        print("MISSING in new:", n); bad += 1; continue
    x, y = pd.read_csv(a / n), pd.read_csv(b / n)
    ok = x.shape == y.shape and x.drop(columns=["area", "max_height"]).equals(y.drop(columns=["area", "max_height"]))
    if ok and len(x):
        ok = np.abs(x.max_height - y.max_height).max() <= 0.11 and np.abs(x.area - y.area).max() <= 2
    if not ok:
        bad += 1; print("DIFFERENT:", n, x.shape, y.shape)
print(f"{len(names)} recordings compared, {bad} different")
