# Handoff: preprocessing -> detection / clustering

Branch: `claude/zealous-cori-9k15vz`. Quick start is in `DEMO.md` (about 1 minute, runs on the processed files).

## Split of work

| Stage | Owner | Where in the code |
|---|---|---|
| Decimation 50 kHz -> 1 kHz, noise reduction (median + Savitzky-Golay) | Preprocessing | `idp/method.py: clean`, `idp/preprocessing.py`, `idp/fast.py: convert` |
| Baseline correction (ALS, rolling median) | Preprocessing | `idp/method.py: als, baseline`, `idp/fast.py: _baseline` |
| Threshold / noise floor (sigma of quietest 10% of 200 ms blocks, k x sigma, no-dye level 69) | Preprocessing | `idp/method.py: quiet_sigma`, `idp/fast.py: _sigma` |
| Peak detection, validation (dyed / solid_no_dye / water, two-channel check) | Detection | `idp/method.py: detect, pieces`, `idp/fast.py: detect` |
| Clustering (peaks < 1 s apart = one piece) | Clustering | `idp/method.py: pieces`, `idp/fast.py: detect` |
| Features and calibration vs volume | shared | `idp/features.py`, `scripts/demo.py`, `results/` |

## Interface between the two halves
Preprocessing output: 1 kHz, baseline-corrected `adc1` and `adc2` (`int16` `.bin`, shape `[n, 2]`, or the `.npz` from `idp/preprocessing.py`) plus per-channel sigma.
Detection input: that signal and sigma; output: pieces with `start_s, n_peaks, max_height, area, label`.

## Results so far
- Detection on all 28 recordings: about 0.3 s (the one-time CSV conversion takes a few minutes).
- HIGH concentration: sum of peak heights vs volume x (1 + re-suctions) fits a line, R2 = 0.996; leave-one-out error +10 / -8 / -5 / -10 / -25 %.
- Big pieces: 41 dyed pieces with the slide-method settings (slide says 41).

## Open items
Preprocessing:
- Slide parameters I had to guess: median 3, Savitzky-Golay 7 (order 3), ALS lam 1e5 / p 0.01, k = 30. Replace with the real values.
- The raw sampling rate (50 kHz) is assumed, not confirmed.
- k and the no-dye level (69) are fixed; derive them per file from blank segments.
- Noise floor varies between files (MAD 2.8 to 7.8 on adc2).

Detection / clustering:
- A 1 s gap rule does not give the number of suctions: the 4.7 h run (1 suction + 3 re-suctions) gives 255 pieces, while its peaks form 4 clear groups many minutes apart. A larger gap, or a second clustering level, is needed.
- No ground-truth peak locations, so false-positive / false-negative rates are not measured.
- MIDDLE concentration does not calibrate well (R2 0.93 on peak heights, worse on piece counts).
- Tumor files with re-suction do not scale monotonically with size.
