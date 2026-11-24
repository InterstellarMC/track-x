# Where signal identity changes

The first change is the first outward hit with a nonzero ID different from the first signal ID. It identifies a discrepancy, not which particle owned the intended trajectory. Noise hits alone do not count as a second signal. Exact retention divides unchanged all-one-signal baseline seeds by all such originals. Seed purity is the largest signal contribution divided by seed length, averaged over all seeds. Seed completeness divides that contribution by all volume-8 hits of that particle. All-noise seeds contribute zero. Seed efficiency counts uniquely matched pixel particles with ≥3 hits, strict majority purity and ≥50% pixel completeness.

| event | pixel occupancy | baseline seeds | mixed at seed | mixed final paths | fraction already mixed | first divergent layer counts |
|---|---|---|---|---|---|---|
| event000001003 | 24979 | 4324 | 1395 | 1598 | 0.872966 | {'8': 110, '4': 838, '6': 447} |
| event000001004 | 33799 | 5828 | 2244 | 2500 | 0.897600 | {'8': 149, '4': 1515, '6': 580} |
| event000001005 | 26241 | 4504 | 1506 | 1717 | 0.877111 | {'8': 117, '4': 935, '6': 454} |
| event000001002 | 30459 | 5261 | 1886 | 2146 | 0.878844 | {'6': 535, '4': 1236, '8': 115} |

| baseline stratum | seeds (development) | mixed fraction |
|---|---|---|
| abs_half_curvature/0 | 3408 | 0.538732 |
| abs_half_curvature/1 | 10133 | 0.289549 |
| abs_half_curvature/2 | 1115 | 0.336323 |
| cross_sector/0 | 10183 | 0.364038 |
| cross_sector/1 | 4473 | 0.321484 |
| length/3 | 6199 | 0.540087 |
| length/4 | 8457 | 0.212487 |
| majority_pt/0 | 7291 | 0.515979 |
| majority_pt/1 | 4794 | 0.190655 |
| majority_pt/2 | 1986 | 0.181772 |
| majority_pt/3 | 576 | 0.187500 |
| majority_pt/noise | 9 | 0.000000 |
| residual/0 | 9256 | 0.063959 |
| residual/1 | 1389 | 0.606911 |
| residual/2 | 1751 | 0.902342 |
| residual/3 | 2260 | 0.942478 |
| skipped_pixel_layer/0 | 12929 | 0.280919 |
| skipped_pixel_layer/1 | 1727 | 0.876086 |

Bins use |half-curvature| boundaries 0.25, 1 and 3 per metre; residual boundaries 0.5, 1 and 2; dominant particle pT boundaries 0.5, 1 and 2 GeV. The sector flag describes crossing an unshifted seam. The baseline reconciles complete paths from two tilings without splicing paths. A skipped pixel layer can arise from missing detections or pattern recognition. These overlapping diagnostics do not identify unique causes. Nearby measurements and ownership conflicts are directly tested by replacement and joint selection.

On validation event 1002, the truth-free maximum seed residual had ROC AUC 0.9544 and average precision 0.9280 for detecting mixed seeds at prevalence 0.3585. These are ranking statistics for an untrained scalar predictor. A good detector of bad seeds can still make a poor rejection policy. Replacement converted 52 mixed seeds to single-signal or noise-only paths and introduced mixing in 8 cases. Other changed seeds retained their mixing status. The reference scorer alone took 0.250 s on this validation run. [Diagnostic record](seed-detection.json)
