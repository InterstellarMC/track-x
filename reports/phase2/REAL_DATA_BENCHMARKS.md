# Official TrackML simulation benchmarks

| builder | method | official_full_event | barrel_efficiency | full_efficiency | fake_rate | edge_ap |
|---|---|---|---|---|---|---|
| original | classical | 0.00300 | 0.13761 | 0.00523 | 0.82918 | 0.16496 |
| original | mlp | 0.00233 | 0.11055 | 0.00415 | 0.86395 | 0.15509 |
| original | gnn | 0.00195 | 0.09678 | 0.00356 | 0.88839 | 0.13497 |
| original | hybrid | 0.00189 | 0.09849 | 0.00355 | 0.88290 | 0.13497 |
| directional | classical | 0.00534 | 0.23722 | 0.01130 | 0.76460 | 0.19330 |
| directional | mlp | 0.00271 | 0.13754 | 0.00571 | 0.85758 | 0.14382 |
| directional | gnn | 0.00247 | 0.13290 | 0.00547 | 0.86258 | 0.14299 |
| directional | hybrid | 0.00249 | 0.13553 | 0.00543 | 0.85645 | 0.14299 |
| directional | local_mlp | 0.00028 | 0.02883 | 0.00109 | 0.90982 | 0.08987 |
| directional | local_small | 0.00298 | 0.22534 | 0.00918 | 0.72917 | 0.22326 |
| directional | local_gnn | 0.00019 | 0.02578 | 0.00113 | 0.81804 | 0.08908 |
| directional | local_layer | 0.00038 | 0.03100 | 0.00125 | 0.83738 | 0.09183 |
| directional | local_hybrid | 0.00021 | 0.02600 | 0.00114 | 0.80489 | 0.08908 |
| directional | local_hybrid_high | 0.00000 | 0.00000 | 0.00000 | undefined | 0.08908 |

All rows are equal-weight means across three complete held-out events. official_full_event calls the pinned organizers’ score_event on every hit. Unused hits share track ID zero, making one explicit background candidate. Only a strict majority in both candidate and full truth particle earns its correctly assigned weight, normalized by event weight. Full-event efficiency uses every truth particle with at least three hits. The barrel overlap metrics use eligible volume-8 particles with at least three barrel hits. Their rule requires >50% purity and ≥50% completeness. Each truth particle can be claimed once. Extra qualifying candidates are duplicates; remaining unmatched candidates are fakes. Their denominators are all reconstructed candidates. Undefined event ratios are omitted from means.

Edge AP, precision, recall and F1 refer to candidate links only. PR uses sigmoid scores or geometric ranking. Scores are uncalibrated; conditioning and matching rules differ across endpoints. See the event JSON for individual measurements, PR samples, fit residual quantiles and denominator counts. Whole-path reconstruction results follow.

| method | sectors | drop | curvature | official_score | efficiency | fake | graph_s | assembly_s | efficiency_event_SD |
|---|---|---|---|---|---|---|---|---|---|
| edge | 32 | 0.00000 | 1.50000 | 0.00021 | 0.02600 | 0.80489 | 0.11600 | 0.06050 | 0.00483 |
| triplet | 32 | 0.00000 | 1.50000 | 0.01552 | 0.48630 | 0.48550 | 0.11115 | 0.37420 | 0.02003 |
| triplet_ml | 32 | 0.00000 | 1.50000 | 0.01546 | 0.48664 | 0.48537 | 0.10153 | 0.38030 | 0.01961 |
