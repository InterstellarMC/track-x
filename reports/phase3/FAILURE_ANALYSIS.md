# Tracking failure diagnostics

| event | method | two_particle_merge_fraction | any_contamination_fraction | fraction_rms_over_5mm |
|---|---|---|---|---|
| event000001009 | seed | 0.00000 | 0.33111 | 0.00000 |
| event000001009 | paths | 0.00289 | 0.41400 | 0.14044 |
| event000001010 | seed | 0.00000 | 0.33599 | 0.00000 |
| event000001010 | paths | 0.00258 | 0.41799 | 0.14316 |
| event000001011 | seed | 0.00000 | 0.33827 | 0.00000 |
| event000001011 | paths | 0.00308 | 0.41888 | 0.14847 |

A substantial two-particle merge requires at least two hits from each of at least two distinct nonzero particles. Any contamination requires just one hit from a second particle. The fit flag uses an empirical 5 mm RMS threshold on tangential and longitudinal residuals from the final prompt fit. It is a diagnostic, not a calibrated rejection cut. Full residual quantiles are in fit-quality.json.

| event | extendible_majority_seeds | no_correct_extension | incorrect_assignment_seed | missing_nominal_surface | missed_and_missing_surface | seeds_pt_below_0.5 | missed_pt_below_0.5 | pixel_seed_spans_original_seam | no_signal_seed_majority | available_13 | missed_13 | available_17 | missed_17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event000001009 | 2984 | 125 | 464 | 2049 | 93 | 1174 | 57 | 987 | 1296 | 2984 | 125 | 1311 | 133 |
| event000001010 | 2815 | 117 | 503 | 1922 | 101 | 1153 | 68 | 939 | 1273 | 2815 | 117 | 1247 | 155 |
| event000001011 | 3158 | 111 | 498 | 2115 | 89 | 1233 | 54 | 1045 | 1468 | 3158 | 111 | 1399 | 119 |

| event | candidate_recall | added | wrong_outer | assessable_outer | mixed | duplicate |
|---|---|---|---|---|---|---|
| event000001009 | 0.96120 | 14625 | 1066 | 11599 | 0.41400 | 0.00022 |
| event000001010 | 0.95799 | 13843 | 1068 | 10918 | 0.41799 | 0.00023 |
| event000001011 | 0.96940 | 15904 | 1152 | 12411 | 0.41888 | 0.00021 |

Candidate recall divides the number of seed/surface opportunities with at least one correct proposed hit by all opportunities for majority-signal seeds with a truth hit on that surface, including terminated seeds. Accepted wrong-outer counts include noise. Missed transition counts refer to extendible seeds without a correct hit in the given volume. Sparse trajectories and low-pT groups can overlap. Seam counts identify seeds crossing an original sector boundary; the one-pass comparison measures the net effect. The straight and simple_fit controls diagnose sensitivity to propagation errors. These overlapping indicators do not establish exclusive causal fractions.
