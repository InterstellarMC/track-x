# Matched ablations

| builder | method | official_full_event | barrel_efficiency | full_efficiency | fake_rate | edge_ap |
|---|---|---|---|---|---|---|
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

Within the directional graph, local_mlp removes message updates, local_small uses 16 hidden units, local_gnn uses 32, and local_layer adds a learned four-layer embedding. All train on identical events with the same seed, optimizer and epoch budget. Minimum unweighted validation BCE selects each checkpoint. Weighted training scores at threshold 0.5 can produce poor path efficiency. local_hybrid_high raises the gate to 0.7; a missing fake rate means there were no candidates. The embedding and width comparisons also change parameter count. A detector-volume embedding cannot vary within this single-volume reconstruction.

| method | sectors | drop | curvature | official_score | efficiency | fake | graph_s | assembly_s | efficiency_event_SD |
|---|---|---|---|---|---|---|---|---|---|
| edge | 32 | 0.00000 | 1.50000 | 0.00021 | 0.02600 | 0.80489 | 0.11600 | 0.06050 | 0.00483 |
| triplet | 32 | 0.00000 | 1.50000 | 0.01552 | 0.48630 | 0.48550 | 0.11115 | 0.37420 | 0.02003 |
| triplet_ml | 32 | 0.00000 | 1.50000 | 0.01546 | 0.48664 | 0.48537 | 0.10153 | 0.38030 | 0.01961 |
| triplet_ml | 32 | 0.10000 | 1.50000 | 0.01217 | 0.40247 | 0.52433 | 0.14473 | 0.33383 | 0.01769 |
| triplet_ml | 16 | 0.00000 | 1.50000 | 0.01781 | 0.56327 | 0.42110 | 0.10172 | 0.39900 | 0.02584 |
| triplet_ml | 64 | 0.00000 | 1.50000 | 0.01133 | 0.33417 | 0.62377 | 0.18051 | 0.49328 | 0.00658 |
| triplet_ml | 32 | 0.00000 | 0.00000 | 0.01198 | 0.31666 | 0.67238 | 0.15723 | 0.48360 | 0.02087 |

triplet uses whole-path geometric ranking. triplet_ml breaks geometric ambiguities with an added −0.2 log product of edge scores. Both prefer compatible four-hit hypotheses to triples before exclusive assignment. The edge method shares the same local GNN weights, but uses thresholded greedy links. Curvature is the coefficient of the radial-gap term in the search window, not fitted truth curvature. The missing-hit study removes observations before reconstruction and scores against the original unthinned truth.
