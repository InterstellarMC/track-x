# Apple M1 CPU resource measurements

| event | load_validate_s | spatial_index_s | peak_MB | graph_s | seed_assembly_s | extension_s | fit | search | ownership |
|---|---|---|---|---|---|---|---|---|---|
| event000001002 | 0.37601 | 0.02854 | 347.81594 | 0.49331 | 1.89704 | 0.24939 | 0.07868 | 0.09504 | 0.07088 |
| event000001003 | 0.33248 | 0.02663 | 347.81594 | 0.36085 | 1.51049 | 0.24391 | 0.08769 | 0.09203 | 0.05896 |
| event000001004 | 0.55528 | 0.04586 | 347.81594 | 0.34474 | 1.73578 | 0.25422 | 0.08109 | 0.09956 | 0.06818 |
| event000001005 | 0.41696 | 0.03856 | 347.81594 | 0.38539 | 1.38520 | 0.18202 | 0.06126 | 0.06978 | 0.04456 |
| event000001009 | 0.30200 | 0.02731 | 376.56986 | 0.22543 | 0.92905 | 0.13752 | 0.04515 | 0.05329 | 0.03636 |
| event000001010 | 0.24677 | 0.01765 | 376.56986 | 0.14310 | 0.67701 | 0.12617 | 0.04152 | 0.04944 | 0.03224 |
| event000001011 | 0.25753 | 0.01863 | 376.56986 | 0.17371 | 0.93009 | 0.16122 | 0.05120 | 0.06305 | 0.04142 |

Process high-water RSS includes earlier event work in the process. Reconstruction time includes indexing, pixel graph, pixel assembly and extension. Loading, official scoring and diagnostic work are timed separately or excluded. Timings are single-run measurements. All seed construction uses one torch CPU thread; layer queries use one worker. At most eight cross-surface proposals per active seed enter ranking on each layer.
