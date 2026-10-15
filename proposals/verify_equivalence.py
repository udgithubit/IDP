"""Check that fast_core gives the same pieces as the group's Part A / Part B, and time both.

    python verify_equivalence.py GROUP_REPO_DIR FILE_OR_FOLDER [FILE_OR_FOLDER ...]

GROUP_REPO_DIR is your clone of IDP2026_group8 (the folder that contains core/). Folders are searched
for files over 1 MB (any extension). Recordings are opened through the group's cache, so the first
run per file also reads the CSV; later runs are fast. Nothing is written into the repository.
"""
import sys, time
from pathlib import Path
import numpy as np

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import pipeline, preprocessing as P, detection as D     # the group's code
import fast_core as F

files = []
for a in sys.argv[2:]:
    p = Path(a)
    files += sorted(f for f in p.rglob("*") if f.is_file() and f.stat().st_size > 1_000_000 and f.suffix.lower() not in (".7z", ".zip", ".png", ".odt", ".json", ".txt")) if p.is_dir() else [p]
files = [f for f in files if "copy" not in f.name.lower()]

print(f"{'recording':44} {'A old':>6} {'A new':>6} {'B old':>6} {'B new':>6}  identical")
tot = np.zeros(4); bad = 0
for f in files:
    idx, summ, _ = pipeline.open_recording(f, pipeline.DEFAULT_CACHE); m = summ["mean"]
    t = time.perf_counter(); c0 = P.preprocess(m); a0 = time.perf_counter() - t
    t = time.perf_counter(); c1 = F.preprocess_fast(m); a1 = time.perf_counter() - t
    t = time.perf_counter(); p0, _ = D.find_pieces(c0); b0 = time.perf_counter() - t
    orig = D.detect_peaks; cache = {}
    def patched(x, s, fs, k=D.K_SIGMA): return cache[id(x)]
    t = time.perf_counter()
    cache[id(c1["adc1"])], cache[id(c1["adc2"])] = F.detect_both(c1["adc1"], c1["sigma_adc1"], c1["adc2"], c1["sigma_adc2"], c1["fs"], D.K_SIGMA, orig)
    D.detect_peaks = patched
    try: p1, _ = D.find_pieces(c1)
    finally: D.detect_peaks = orig
    b1 = time.perf_counter() - t
    same = (p0.shape == p1.shape and p0.drop(columns=["area", "max_height"]).equals(p1.drop(columns=["area", "max_height"]))
            and (len(p0) == 0 or (np.abs(p0.max_height.values - p1.max_height.values).max() <= 0.11 and np.abs(p0.area.values - p1.area.values).max() <= 2)))
    bad += not same; tot += [a0, a1, b0, b1]
    print(f"{f.name[:44]:44} {a0:6.2f} {a1:6.2f} {b0:6.2f} {b1:6.2f}  {'yes' if same else 'NO'} ({len(p0)} pieces)", flush=True)
print(f"\n{len(files)} recordings | Part A {tot[0]:.1f} -> {tot[1]:.1f} s | Part B {tot[2]:.1f} -> {tot[3]:.1f} s | not identical: {bad}")
