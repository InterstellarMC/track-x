# Full-event computational measurements

Single-process CPU evaluation. RSS is the process lifetime high-water mark including all evaluated methods, libraries and scoring. Timing components can vary with thermal state and other processes. CPU covers seeding and extension. Wall excludes CSV loading, scoring and diagnostic reductions.

| event | hits | CSV validation/loading s | outer index s | peak MiB |
|---|---|---|---|---|
| event000001022 | 104514 | 0.259170 | 0.022431 | 350.750000 |
| event000001023 | 128037 | 0.472333 | 0.040199 | 350.750000 |
| event000001024 | 117272 | 0.253569 | 0.018684 | 350.750000 |
| event000001025 | 119916 | 0.349212 | 0.027429 | 350.750000 |
| event000001026 | 122690 | 0.463310 | 0.035869 | 350.750000 |

| method | wall s | CPU s | graph s | seed assembly s | evaluation s | extension stage means (s) |
|---|---|---|---|---|---|---|
| M0 | 1.878690 | 1.886478 | 0.276223 | 1.304828 | 0.796708 | fit 0.0986, search 0.0939, ownership 0.0707 |
| M1 | 1.746463 | 1.810895 | 0.276223 | 1.304828 | 0.952596 | fit 0.0540, search 0.0422, ownership 0.0363 |
| M2 | 1.949355 | 1.989019 | 0.276223 | 1.304828 | 0.827155 | search 0.2907, update 0.0020, ownership 0.0341 |
| M3 | 1.897135 | 1.941289 | 0.276223 | 1.304828 | 1.097379 | search 0.2101, update 0.0299, ownership 0.0364 |
| M4 | 2.011809 | 2.059150 | 0.276223 | 1.304828 | 0.747761 | search 0.2679, update 0.0022, ownership 0.0432, global 0.0760 |
| M5 | 2.103614 | 2.039161 | 0.276223 | 1.304828 | 0.992923 | search 0.1970, update 0.0313, ownership 0.0326, global 0.2198 |
| M6 | 2.642818 | 2.591007 | 0.268186 | 2.039372 | 0.981263 | search 0.2013, update 0.0282, ownership 0.0325 |

## Fresh-directory replay

| method | confirmation wall s/event | replay wall s/event |
|---|---|---|
| M0 | 1.878690 | 1.225178 |
| M1 | 1.746463 | 1.172514 |
| M2 | 1.949355 | 1.294236 |
| M3 | 1.897135 | 1.266776 |
| M4 | 2.011809 | 1.377726 |
| M5 | 2.103614 | 1.343532 |
| M6 | 2.642818 | 1.984984 |

Replay peak process RSS was 398.6 MiB. Session load varied and no exclusive-core or thermal controls were used. These two measurements show the variation in absolute timing.
