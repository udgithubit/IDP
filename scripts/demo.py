"""Live demo: python scripts/demo.py BIN_DIR [OUT_PNG]
Detect peaks/pieces in every preprocessed recording, label them, estimate volume from the
calibration line, and report timing. Needs the .bin files made by scripts/run_fast.py.
"""
import sys, time, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from idp import fast

bins = Path(sys.argv[1]); png = sys.argv[2] if len(sys.argv) > 2 else "demo.png"
files = sorted(bins.glob("*.bin"))
t0 = time.perf_counter(); res = {}
for f in files:
    t = time.perf_counter(); r = fast.detect(f); r["ms"] = (time.perf_counter() - t) * 1e3; res[f.stem] = r
total = time.perf_counter() - t0
print(f"\nDetected peaks in {len(files)} recordings in {total:.2f} s "
      f"({sum(r['peaks'] for r in res.values()):,} peaks, "
      f"{sum(f.stat().st_size for f in files)/1e6:.0f} MB on disk)\n")

# calibration on HIGH known-volume files: sum_height = slope * (volume_ml * (1 + re-suctions))
units = {"known_1": 16, "known_2": 12, "known_3": 6, "known_4": 3, "known_5": 1}
def key(n): return n.split(".")[0].split("_suction")[0] if n.startswith("known_") else n
cal = {k: (u, next(v for n, v in res.items() if n.startswith(k + ".") or n.startswith(k + "_"))) for k, u in units.items()}
u = np.array([c[0] for c in cal.values()], float); h = np.array([c[1]["sum_height"] for c in cal.values()])
slope = (u * h).sum() / (u * u).sum()
print(f"Calibration (HIGH conc): sum of peak heights = {slope:,.0f} x (volume_ml x (1 + re-suctions))")
print(f"{'recording':58} {'ms':>6} {'peaks':>6} {'pieces':>6} {'dyed':>5} {'solid':>5} {'water':>5} {'est.units':>9}")
for n, r in res.items():
    est = r["sum_height"] / slope
    print(f"{n[:58]:58} {r['ms']:6.1f} {r['peaks']:6d} {r['pieces']:6d} {r['dyed']:5d} {r['solid_no_dye']:5d} {r['water']:5d} {est:9.1f}")
print("\nLeave-one-out error on the 5 HIGH known-volume files:")
for i, k in enumerate(units):
    m = np.arange(5) != i; s = (u[m] * h[m]).sum() / (u[m] ** 2).sum()
    print(f"  {k}: true {u[i]:>4.0f} units, predicted {h[i]/s:5.1f}  ({100*(h[i]/s-u[i])/u[i]:+.0f}%)")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
k1 = next(n for n in res if n.startswith("known_1.") or n.startswith("known_1_"))
r = res[k1]; ax[0].plot(r["peak_t"] / 60, r["peak_h"], ".", ms=2); ax[0].set(title="4.7 h run: detected peaks", xlabel="min", ylabel="height (counts)")
ax[1].plot(u, h / 1e3, "o"); xs = np.linspace(0, 17, 2); ax[1].plot(xs, slope * xs / 1e3, "-")
ax[1].set(title="Calibration (HIGH)", xlabel="volume_ml x (1 + re-suctions)", ylabel="sum of peak heights (k counts)")
ms = [r["ms"] for r in res.values()]; ax[2].barh(range(len(ms)), ms); ax[2].set(title=f"Detection time per file (total {total:.2f} s)", xlabel="ms", yticks=[])
plt.tight_layout(); plt.savefig(png, dpi=110); print(f"\nFigure saved: {png}")
