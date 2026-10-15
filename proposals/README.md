# Speed-up proposal for Part A / Part B of the group pipeline (results identical)

`fast_core.py` holds drop-in replacements. Nothing here changes results: on all 28 recordings the pieces
table (labels, n_peaks, start/end) is identical, max_height differs by <= 0.1 count, area by <= 2 counts-ms,
sigma by 1e-7 relative.

| Stage (7.6 GB / 4.7 h recording, this sandbox) | old | new |
|---|---|---|
| Part A, both channels | 9.7 s | 2.8 s (3.5x) |
| Part B, two find_peaks calls in parallel threads | 6.2 s | 3.4 s (1.8x) |
| Part A + B | 15.9 s | 6.2 s |

What changed
- `median3`: vectorised min/max instead of `scipy.signal.medfilt` (same values; zero padding at the edges like medfilt).
- `als_banded`: banded Cholesky (`solveh_banded`) instead of `sparse.diags` + `spsolve` (ALS 1.9 s -> 0.57 s).
- Both channels cleaned in two threads (`preprocess_fast`), and the adc1 / adc2 `find_peaks(prominence)` calls in two threads (`detect_both`).
- float64 is kept: a float32 version was faster but shifted sigma by 0.15% and changed one piece's peak count.

Tried and dropped
- `find_peaks(prominence)` only near activity: exact only with windows >= 10 s, which saves nothing; with 1 s windows it missed 82 peaks.

How to use (for core/preprocessing.py and core/detection.py)
1. In `preprocessing.preprocess`, call `preprocess_fast(mean_1khz, fs, params)` instead of the two `clean_channel` calls.
2. In `detection.find_pieces`, replace the two `detect_peaks` calls by
   `p1, p2 = detect_both(a1, sigma_adc1, a2, sigma_adc2, fs, k, detect_peaks)` (adc1 only if `use_adc1`).

Not done (bigger, but in fast_io): the row index (6.6 s) and the 1 kHz summary (10 s with numba, 21 s without) each
read the whole CSV; a single pass, or skipping the index when only the summary is needed, would save about 6.6 s.
