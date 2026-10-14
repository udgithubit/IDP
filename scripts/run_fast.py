"""python scripts/run_fast.py DATA_DIR BIN_DIR : convert every CSV once, then time detection."""
import sys, glob, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from idp import fast
data, out = Path(sys.argv[1]), Path(sys.argv[2])
rows, tconv = [], 0.0
for f in sorted(glob.glob(str(data / "*" / "*.csv"))):
    g = Path(f).parent.name; name = f"{g}_{Path(f).stem}"
    b = out / f"{name}.bin"
    if not b.exists():
        t = time.time(); fast.convert(f, out); tconv += time.time() - t
        os.replace(out / (Path(f).stem + ".bin"), b); os.replace(out / (Path(f).stem + ".json"), out / f"{name}.json")
    rows.append({"file": name, "bin_kb": b.stat().st_size / 1e3})
t0 = time.time()
for r in rows:
    t = time.time(); r.update(fast.detect(out / f"{r['file']}.bin")); r["detect_s"] = time.time() - t
tdet = time.time() - t0
df = pd.DataFrame(rows); df.to_csv(out / "fast_results.csv", index=False)
print(df.round(3).to_string(index=False))
print(f"\nconvert (one-time) {tconv:.0f}s | detect all {len(rows)} files {tdet:.2f}s | storage {df.bin_kb.sum()/1e3:.1f} MB")
