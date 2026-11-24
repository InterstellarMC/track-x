# Seed residual rejection and same-layer replacement

Given a frozen three/four-hit seed, compare its measured positions to a robust transverse prompt circle and an arc-length z regression. Use the largest normalized 2D residual, with effective scales 0.35 mm tangential and 0.7 mm longitudinal. The strict arm discards seeds above two units. Replacement considers one same-layer substitution at a time among four spatial neighbors inside 8 mm, including the hit itself. Each original plus two lowest-cost alternatives reaches assignment. All variants preserve radial and layer order.

| method | score | seed_purity | retention | efficiency | mixed |
|---|---|---|---|---|---|
| geometry | 0.220071 | 0.772075 | 1.000000 | 0.258915 | 0.430569 |
| local | 0.223755 | 0.772075 | 1.000000 | 0.263651 | 0.393249 |
| global | 0.223589 | 0.772075 | 1.000000 | 0.263320 | 0.382836 |
| tight | 0.222055 | 0.772075 | 1.000000 | 0.262005 | 0.378905 |
| strict | 0.220206 | 0.843858 | 0.997936 | 0.256036 | 0.291632 |
| replace | 0.224635 | 0.774036 | 0.895742 | 0.264651 | 0.386733 |
| beam | 0.224720 | 0.774335 | 0.897439 | 0.264872 | 0.386599 |
| joint | 0.225026 | 0.787179 | 0.792088 | 0.264808 | 0.363818 |

Retention is intentionally exact. A genuine same-particle replacement also changes the seed identity under this test.
