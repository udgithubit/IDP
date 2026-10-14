# Results (adc2, 1 kHz max-envelope, k=5 MAD noise floor, min 15 counts)

Calibration on the known-volume set (feature = sum of peak heights):

- HIGH conc: sum_height ≈ 58–61k × (volume_ml × (1 + re-suctions)); see fit below (5 files incl. the 4.7 h run)
- MIDDLE conc: ≈ 13.3k per unit, R² = 0.93 (4 files); HIGH/MIDDLE ratio ≈ 4.3
- Tumor files (HIGH): sum_height 1 gr : 2 gr : 3 gr ≈ 1 : 1.9 : 2.9 (no re-suction),
  1 : 1.5 : 2.1 (re-suction); 1 gr has 3 repeats ranging 41k–93k.

Caveats: raw sampling rate assumed 50 kHz (the long run then lasts 4.69 h, consistent with the '4½ h' label); file 1 included (7.6 GB CSV, 4455 peaks, sum_height 980k = 61.3k per unit).

## Slide-deck method (idp/method.py, scripts/run_method.py)

Median(3) -> Savitzky-Golay(7,3) -> ALS baseline; sigma = std of quietest 10% of 200 ms blocks;
peaks > 30 sigma; peaks < 1 s apart = one piece; dyed if max > 69 counts (no-dye level from the
slides), else solid_no_dye if an adc1 peak is within +-20 ms, else water; amount = sum of dyed areas.
Check on Big pieces: 41 dyed pieces (slide: 41), first piece at 122.6 s with area 10982 (slide: 11227).
Per-file output: results/method_results.csv, per-piece output: results/method_pieces.csv.

## Fast path (idp/fast.py, scripts/run_fast.py)

One-time convert (pyarrow CSV -> mean-decimate 50:1 -> baseline subtracted -> int16 .bin),
then detection on the memory-mapped .bin skipping quiet 100 ms blocks.
Measured here (4 cores): convert all 28 CSVs once 158 s (7.6 GB file ~70-100 s);
detect all 28 files 0.57 s (7.6 GB file 0.375 s); storage 97 MB (67 MB is the 4.7 h file).
