# Speed-up proposal for Part A / Part B of the group pipeline (results identical)

`fast_core.py` holds drop-in replacements. Nothing here changes results: on all 28 recordings the pieces
table (labels, n_peaks, start/end) is identical, max_height differs by <= 0.1 count, area by <= 2 counts-ms,
sigma by 1e-7 relative.

| Stage, 7.6 GB / 4.7 h recording | old | new |
|---|---|---|
| Part A, both channels (Sora, Mac, 32 recordings: 2.7x over all) | 1.71 s | 0.62 s |
| Part B, two find_peaks calls in parallel threads (Sora, Mac) | 1.72 s | 1.02 s |
| Part A, both channels (cloud container, 4 cores) | 9.7 s | 2.8 s |
| Part B, same change (cloud container) | 6.2 s | 3.4 s |
| Part A (Unmona laptop, no Mac) | about 3.5 s | about 2.0 s |

Absolute numbers differ a lot between machines; the ratios are what to compare.

What changed
- `median3`: vectorised min/max instead of `scipy.signal.medfilt` (same values; zero padding at the edges like medfilt).
- `als_banded`: banded Cholesky (`solveh_banded`) instead of `sparse.diags` + `spsolve` (ALS 1.9 s -> 0.57 s).
- Both channels cleaned in two threads (`preprocess_fast`), and the adc1 / adc2 `find_peaks(prominence)` calls in two threads (`detect_both`).
- sigma and adc1_response are computed from the float64 signal before the cast to float32, like core (review by Sora). The sigma difference to core is below 1e-10 relative (banded vs sparse solver rounding), not bit-identical.
- `median3` only handles size 3 (other sizes fall back to `medfilt`); `als_banded` needs n >= 4 points (shorter input: no baseline).
- float64 is kept: a float32 version was faster but shifted sigma by 0.15% and changed one piece's peak count.

Tried and dropped
- `find_peaks(prominence)` only near activity: exact only with windows >= 10 s, which saves nothing; with 1 s windows it missed 82 peaks.

How to use (for core/preprocessing.py and core/detection.py)
1. In `preprocessing.preprocess`, call `preprocess_fast(mean_1khz, fs, params)` instead of the two `clean_channel` calls.
2. In `detection.find_pieces`, replace the two `detect_peaks` calls by
   `p1, p2 = detect_both(a1, sigma_adc1, a2, sigma_adc2, fs, k, detect_peaks)` (adc1 only if `use_adc1`).

Not done (bigger, but in fast_io): the row index (6.6 s) and the 1 kHz summary (10 s with numba, 21 s without) each
read the whole CSV; a single pass, or skipping the index when only the summary is needed, would save about 6.6 s.


## core/preprocessing.py
`core_preprocessing.patch` (unified diff against the group's `core/preprocessing.py`) and `core_preprocessing_new.py` (the whole file) move Part A into core: `git apply proposals/core_preprocessing.patch`. Checked on 28 recordings: pieces and labels identical, sigma within 3e-11 relative.
