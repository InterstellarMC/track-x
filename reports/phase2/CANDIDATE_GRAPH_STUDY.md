# Paired candidate study

Denominator includes every same, nonzero particle pair on volume-8 layers one or two transitions apart across the entire barrel, including pairs separated by sector boundaries. Multiple module hits contribute multiple possible pairs. This is an available association graph, not a unique chronological trajectory. Pairs on a curling return can be counted even when outward tracking is inappropriate. No momentum, truth or labels enter either builder.

The directional search takes at most eight target hits per layer in scaled (cos φ, sin φ, z/r). Its azimuth scale increases with measured layer separation; its longitudinal tolerance reflects beamline vertex spread. Both builders share partitions and graph features. Directed density divides by N(N−1) across barrel hits, including cross-sector pairs.

| builder | truth_pairs | edges | false_candidates | recall | edges_per_hit | directed_density |
|---|---|---|---|---|---|---|
| directional | 34218.00000 | 259688.66667 | 241401.00000 | 0.53725 | 9.14996 | 0.00033 |
| original | 34218.00000 | 306759.00000 | 293163.66667 | 0.39961 | 10.81747 | 0.00039 |

| builder | neighbor_cap | boundary | longitudinal | kept | direction_window |
|---|---|---|---|---|---|
| directional | 16734 | 29720 | 0.00000 | 54863 | 1337.00000 |
| original | 10608 | 29720 | 21540.00000 | 40786 | 0.00000 |

Reason counts pool all held-out eligible pairs. A boundary failure takes precedence over geometric rejection. neighbor_cap counts in-window, in-sector outward pairs excluded by candidate ranking. More detailed occupancy bins are in candidate-losses.json.

Averaging is per held-out event. Transition counts and sector occupancy are stored per event. Remaining losses include sector seams, finite neighbor budget, projective and curvature windows, and radial-order failures. The separate stress study varies the curvature allowance and sector count.
