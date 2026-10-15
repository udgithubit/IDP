"""Analyse recordings with the group's pipeline, using the faster Part A / Part B from fast_core.

    python run_pipeline_fast.py GROUP_REPO_DIR FILE_OR_FOLDER [FILE_OR_FOLDER ...] [--out DIR] [--workers N]

Same output line as the group's tools/run_pipeline.py. Folders are searched for files over 1 MB
(any extension, so extensionless recordings work); '.7z/.zip/.png/...' and files named '* Copy*' are skipped.
--out writes one pieces table per recording (never put it inside the repository).
"""
import argparse, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo", type=Path); ap.add_argument("paths", type=Path, nargs="+")
ap.add_argument("--out", type=Path, default=None); ap.add_argument("--workers", type=int, default=None)
args = ap.parse_args()
sys.path.insert(0, str(args.repo.resolve())); sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import pipeline, detection as D          # the group's code
import fast_core as F

SKIP = (".7z", ".zip", ".png", ".odt", ".json", ".txt", ".md", ".py")
files = []
for p in args.paths:
    files += sorted(f for f in p.rglob("*") if f.is_file() and f.stat().st_size > 1_000_000 and f.suffix.lower() not in SKIP) if p.is_dir() else [p]
files = [f for f in files if "copy" not in f.name.lower()]
if args.out: args.out.mkdir(parents=True, exist_ok=True)

orig = D.detect_peaks
for f in files:
    t0 = time.perf_counter(); tm = {}
    t = time.perf_counter()
    index, summary, from_cache = pipeline.open_recording(f, pipeline.DEFAULT_CACHE, args.workers)
    tm["read (cache)" if from_cache else "read CSV -> 1 kHz"] = time.perf_counter() - t
    t = time.perf_counter(); clean = F.preprocess_fast(summary["mean"], summary["fs_hz"]); tm["Part A (fast)"] = time.perf_counter() - t
    t = time.perf_counter()
    cache = {id(clean["adc1"]): None, id(clean["adc2"]): None}
    cache[id(clean["adc1"])], cache[id(clean["adc2"])] = F.detect_both(
        clean["adc1"], clean["sigma_adc1"], clean["adc2"], clean["sigma_adc2"], clean["fs"], D.K_SIGMA, orig)
    D.detect_peaks = lambda x, s, fs, k=D.K_SIGMA: cache[id(x)]
    try: pieces, peaks = D.find_pieces(clean)
    finally: D.detect_peaks = orig
    tm["Part B (fast)"] = time.perf_counter() - t
    counts = pieces["label"].value_counts().to_dict() if len(pieces) else {}
    minutes = len(clean["adc2"]) / clean["fs"] / 60
    steps = "  ".join(f"{k} {v:.1f} s" for k, v in tm.items())
    print(f"{f.name[:50]:<52}{minutes:7.1f} min  pieces {len(pieces):4d}  {counts}  sigma {clean['sigma_adc2']:.2f}  "
          f"adc1 {'used' if clean['adc1_usable'] else 'NOT USED'}  |  {steps}  | total {time.perf_counter() - t0:.1f} s", flush=True)
    if args.out: pieces.to_csv(args.out / f"{f.stem}.pieces.csv", index=False)
