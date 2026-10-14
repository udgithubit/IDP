"""Export Part A output per recording: python scripts/export_clean.py NPZ_DIR OUT_DIR
Reads the decimated .npz files (idp/preprocessing.py), applies median + Savitzky-Golay + ALS
baseline correction, estimates sigma, and writes <name>.clean.npz for the detection stage."""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from idp import method as M

npz_dir, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for f in sorted(glob.glob(str(npz_dir / "*.npz"))):
    n = Path(f).stem
    if n[0].isdigit():            # skip duplicate un-prefixed files
        continue
    d = np.load(f); fs = float(d["fs"])
    sig = M.run_signal(d["adc1_mean"], d["adc2_mean"], fs)
    np.savez_compressed(out / f"{n}.clean.npz", fs=fs,
                        adc1_i16=np.clip(np.rint(sig["a1"] * 10), -32768, 32767).astype(np.int16),
                        adc2_i16=np.clip(np.rint(sig["a2"] * 10), -32768, 32767).astype(np.int16), scale=0.1,
                        sigma_adc1=sig["sigma"]["adc1"], sigma_adc2=sig["sigma"]["adc2"],
                        k=M.P["k"], source=n)
    print(n, "sigma", {c: round(v, 3) for c, v in sig["sigma"].items()}, flush=True)
