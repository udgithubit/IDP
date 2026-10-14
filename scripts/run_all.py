"""Preprocess every CSV under DATA_DIR/<group>/ and write results/features.csv.

Usage: python scripts/run_all.py DATA_DIR OUT_DIR
Expects DATA_DIR/{known,demo,tumor}/*.csv (download them from the GitHub releases).
"""
import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from idp.preprocessing import preprocess
from idp.features import file_features

data, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "npz").mkdir(parents=True, exist_ok=True)
rows = []
for grp in ("known", "demo", "tumor"):
    for f in sorted(glob.glob(str(data / grp / "*.csv"))):
        n = Path(f).stem
        npz = out / "npz" / f"{grp}_{n}.npz"
        if not npz.exists():
            preprocess(f, npz)
        rows.append({"group": grp, "file": n, **file_features(npz)})
pd.DataFrame(rows).to_csv(out / "features.csv", index=False)
