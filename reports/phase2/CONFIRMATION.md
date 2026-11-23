# Prospective three-event confirmation

The protocol was committed before the three additional event files were fetched. Both passes and both ranking methods used the same previously frozen weights and geometric settings. Scores are official complete-event scores; efficiency and fake fractions are restricted to volume 8. Times exclude CSV loading and truth evaluation.

| method | passes | official_score | efficiency | fake | duplicate | purity | completeness | reconstruction_s |
|---|---|---|---|---|---|---|---|---|
| geometry | 1 | 0.01633 | 0.50267 | 0.47141 | 0.00035 | 0.97557 | 0.80892 | 0.91492 |
| geometry | 2 | 0.01959 | 0.63415 | 0.32905 | 0.00157 | 0.98447 | 0.81018 | 1.83907 |
| gnn_weighted | 1 | 0.01627 | 0.50303 | 0.47050 | 0.00035 | 0.97541 | 0.80845 | 1.13051 |
| gnn_weighted | 2 | 0.01956 | 0.63416 | 0.32890 | 0.00144 | 0.98428 | 0.81005 | 2.47757 |

Paired two-pass minus one-pass changes for geometry-only reconstruction, by event.

| event | official_score | efficiency | fake |
|---|---|---|---|
| event000001006 | 0.00306 | 0.12354 | -0.13828 |
| event000001007 | 0.00357 | 0.13943 | -0.14753 |
| event000001008 | 0.00313 | 0.13146 | -0.14126 |

Every event improved on both prespecified primary endpoints. The small differences between two-pass geometry and GNN-weighted ranking do not establish a learned advantage. The three non-random consecutive archive events limit external validity. [Frozen protocol](CONFIRMATION_PROTOCOL.md) · [Exact file identities](confirmation-manifest.json) · [Individual records](confirmation.json)
