# Bounded competing seed hypotheses

The beam arm reranks the original and two alternatives by maximum pixel residual plus first outer-layer corroboration. For four nearest volume-13 hits, compute the minimum innovation χ² at the actual radius, cap at 9 and divide by 9. Unreachable extrapolations receive neutral corroboration. Sorting all candidates by cost reconciles collisions on unowned alternatives. Reserved original hits guarantee a fallback for each seed. Every candidate has the same length as its parent. This is a one-step beam, without recursive combinatorial branching.

| method | score | mixed | severe | retention | runtime |
|---|---|---|---|---|---|
| geometry | 0.220071 | 0.430569 | 0.003700 | 1.000000 | 1.427316 |
| local | 0.223755 | 0.393249 | 0.002208 | 1.000000 | 1.600411 |
| global | 0.223589 | 0.382836 | 0.001492 | 1.000000 | 1.540079 |
| tight | 0.222055 | 0.378905 | 0.001714 | 1.000000 | 1.606481 |
| strict | 0.220206 | 0.291632 | 0.002082 | 0.997936 | 1.733490 |
| replace | 0.224635 | 0.386733 | 0.002306 | 0.895742 | 1.755178 |
| beam | 0.224720 | 0.386599 | 0.002325 | 0.897439 | 1.861676 |
| joint | 0.225026 | 0.363818 | 0.001855 | 0.792088 | 1.823810 |
