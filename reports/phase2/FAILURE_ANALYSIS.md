# Failure modes

| event | group | eligible | matched | efficiency |
|---|---|---|---|---|
| event000001003 | spans_sector | 2157 | 401 | 0.18591 |
| event000001003 | sparse | 780 | 250 | 0.32051 |
| event000001003 | crowded | 2913 | 1563 | 0.53656 |
| event000001004 | spans_sector | 2959 | 433 | 0.14633 |
| event000001004 | sparse | 1101 | 297 | 0.26975 |
| event000001004 | crowded | 4549 | 2227 | 0.48956 |
| event000001005 | spans_sector | 2296 | 422 | 0.18380 |
| event000001005 | sparse | 833 | 249 | 0.29892 |
| event000001005 | crowded | 3099 | 1631 | 0.52630 |

| noise_quartile | sectors | mean_noise_fraction | mean_nodes | fakes | candidates | sector_fake_rate |
|---|---|---|---|---|---|---|
| Q1 | 24 | 0.06133 | 1025.95833 | 2072 | 4309 | 0.48085 |
| Q2 | 24 | 0.07335 | 908.87500 | 1768 | 3796 | 0.46575 |
| Q3 | 24 | 0.08332 | 834.95833 | 1609 | 3421 | 0.47033 |
| Q4 | 24 | 0.09627 | 772.66667 | 1437 | 3131 | 0.45896 |

Noise quartiles stratify the observed fraction of particle_id-zero hits after reconstruction. Rates pool unmatched candidates within each sector, using sector-local matching. Occupancy covaries with natural noise; this is descriptive, not a controlled injection study. The local failure display selects the first missed cross-sector particle in integer-ID order on the first test event and draws touching candidates. Run scripts/phase2_display.py to generate it.

Sparse denotes exactly three retained barrel hits. Crowded means at least one hit whose nearest same-layer sector neighbor is from a different truth association within 2 mm. Boundary particles occupy more than one sector. Groups overlap. No truth is used to create candidates or resolve ownership. Figures and per-sector fake counts expose occupancy dependence. Zero denominator is undefined.

Four layers limit completeness under the official full-detector score. Fixed angular sectors split viable tracks. The 16/32/64 sector sweep measures the cost of boundaries together with changes in candidate competition. Exclusivity can suppress duplicates while assigning a hit to the wrong candidate. Fragment counts in stress.json count truth particles represented by two or more hits on multiple output paths. They differ from duplicate counts under the majority matching rule.
