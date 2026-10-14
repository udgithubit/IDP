# Demo (about 1 minute)

```bash
pip install numpy scipy pyarrow matplotlib
unzip data/processed_bins.zip -d bins          # 28 recordings, 13 MB zipped (97 MB raw)
python scripts/demo.py bins results/demo.png   # detects peaks in all files, prints table + figure
```

Expected: all 28 recordings (including the 7.6 GB, 4.7 h run) processed in ~0.3 s.

To rebuild the .bin files from raw CSVs (one-time, ~2.5 min for everything):
`python scripts/run_fast.py DATA_DIR bins`  (DATA_DIR/{known,demo,tumor,big}/*.csv from the GitHub releases)
