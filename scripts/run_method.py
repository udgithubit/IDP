"""Run the slide-deck method on every preprocessed .npz and write results/method_results.csv
and results/method_pieces.csv.   Usage: python scripts/run_method.py NPZ_DIR OUT_DIR
"""
import sys, glob, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from idp import method as M

NO_DYE = 69.0   # counts; "top 0.1% of water peaks" from the slides (water-only recordings)
npz_dir, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rows, prow = [], []
for f in sorted(glob.glob(str(npz_dir / "*.npz"))):
    n = Path(f).stem; d = np.load(f)
    sig = M.run_signal(d["adc1_mean"], d["adc2_mean"], float(d["fs"]))
    pcs, s = M.pieces(sig, NO_DYE)
    dy = [p for p in pcs if p["label"] == "dyed"]
    rows.append({"file": n, "duration_s": len(sig["a2"]) / sig["fs"],
                 "sigma_adc1": sig["sigma"]["adc1"], "sigma_adc2": sig["sigma"]["adc2"],
                 "pieces": len(pcs), **s, "dyed_area": sum(p["area"] for p in dy),
                 "dyed_max_sum": sum(p["max_height"] for p in dy)})
    prow += [{"file": n, **p} for p in pcs]
    print(n, len(pcs), s, flush=True)
pd.DataFrame(rows).to_csv(out / "method_results.csv", index=False)
pd.DataFrame(prow).to_csv(out / "method_pieces.csv", index=False)
