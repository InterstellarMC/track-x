# Matched algorithm comparisons

Development test event means (1003–1005), excluding validation 1002.

| method | score | mixed | total_s |
|---|---|---|---|
| baseline | 0.01930 | 0.34734 | 1.90838 |
| extended | 0.22007 | 0.43057 | 2.17121 |
| gnn_extended | 0.22048 | 0.43140 | 3.02214 |
| middle_only | 0.19314 | 0.42040 | 2.11354 |
| no_gaps | 0.22137 | 0.41270 | 2.14449 |
| one_pass | 0.18107 | 0.56796 | 1.32290 |
| original_one | 0.01552 | 0.49968 | 1.08047 |
| seed_order | 0.21697 | 0.44717 | 2.14803 |
| simple_fit | 0.21121 | 0.45968 | 2.13107 |
| straight | 0.04549 | 0.85522 | 2.29229 |
| strict_bridge | 0.22042 | 0.43019 | 2.20526 |

Frozen confirmation event means (1009–1011). Each row reuses the same hits. gnn_extended uses learned pixel-seed ranking and the same geometric outer extension. The score difference measures only that learned intervention. No trained outer-edge model was evaluated.

| method | score | efficiency | fake | duplicate | purity | completeness | mixed | coverage | total_s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.01858 | 0.03071 | 0.94266 | 0.00000 | 0.98233 | 0.70561 | 0.33512 | 0.14769 | 1.02682 |
| extended | 0.22320 | 0.26644 | 0.50241 | 0.00022 | 0.96029 | 0.73504 | 0.41696 | 0.28194 | 1.18896 |
| gnn_extended | 0.22308 | 0.26644 | 0.50279 | 0.00036 | 0.96036 | 0.73520 | 0.41706 | 0.28210 | 1.88845 |
| middle_only | 0.19496 | 0.26435 | 0.50640 | 0.00014 | 0.96749 | 0.66473 | 0.40587 | 0.25438 | 1.14859 |
| no_gaps | 0.22393 | 0.26835 | 0.49886 | 0.00021 | 0.96696 | 0.73568 | 0.39926 | 0.26167 | 1.17456 |
| one_pass | 0.18407 | 0.21025 | 0.60827 | 0.00008 | 0.95525 | 0.73134 | 0.55634 | 0.27371 | 0.67954 |
| original_one | 0.01442 | 0.02292 | 0.95731 | 0.00000 | 0.97702 | 0.70689 | 0.49108 | 0.14357 | 0.52534 |
| seed_order | 0.22130 | 0.26380 | 0.50758 | 0.00000 | 0.95731 | 0.73424 | 0.43290 | 0.28229 | 1.20129 |
| simple_fit | 0.21390 | 0.25948 | 0.51541 | 0.00022 | 0.94133 | 0.71333 | 0.44552 | 0.27861 | 1.16300 |
| straight | 0.04709 | 0.05398 | 0.89923 | 0.00000 | 0.79417 | 0.64277 | 0.84525 | 0.25479 | 1.17604 |
| strict_bridge | 0.22344 | 0.26676 | 0.50174 | 0.00030 | 0.96132 | 0.73502 | 0.41616 | 0.27974 | 1.17649 |
