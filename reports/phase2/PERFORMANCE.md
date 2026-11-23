# M1 performance

Stage times are seconds per event aggregated across sectors on one PyTorch CPU thread. The benchmark executes alternative methods sequentially. Its complete wall time includes validation, model comparisons, scoring and diagnostics. Graph time is shared across methods and must not be summed over rows. Phase II run timings describe local load; device microbenchmarks warm up separately.

| builder | method | graph | inference | assembly | fit |
|---|---|---|---|---|---|
| directional | classical | 0.21455 | 0.00951 | 0.69276 | 1.13936 |
| directional | hybrid | 0.21455 | 0.76788 | 0.52196 | 0.65713 |
| directional | local_gnn | 0.21455 | 0.50215 | 0.04269 | 0.10740 |
| original | classical | 4.74903 | 0.01463 | 0.44922 | 0.42311 |
| original | hybrid | 4.74903 | 0.53899 | 0.42889 | 0.51382 |

| event | hits | loading_and_validation_s | all_method_wall_s | process_peak_MB |
|---|---|---|---|---|
| event000001003 | 104451 | 0.23731 | 24.26471 | 388.62848 |
| event000001004 | 138568 | 0.35605 | 37.27503 | 388.62848 |
| event000001005 | 110023 | 0.33929 | 30.44638 | 388.62848 |

On 500 matched training-event hit sequences, moving coordinate extraction outside the fit loop changed elapsed time from 0.1683 to 0.0416 seconds. Every longitudinal residual was identical. The training pilot measured 2.807 s with the original builder plus classical assembly and 0.414 s with directional candidates. That comparison changes the graph and therefore the work performed.

| device | sectors | nodes | edges | median_ms | peak_rss_bytes | max_score_delta |
|---|---|---|---|---|---|---|
| cpu | 16 | 1811 | 16464 | 14.33500 | 409010176 | 0.00000 |
| cpu | 32 | 924 | 8484 | 16.64708 | 374407168 | 0.00000 |
| cpu | 64 | 528 | 4860 | 8.52360 | 375783424 | 0.00000 |
| mps | 16 | 1811 | 16464 | 31.33973 | 504676352 | 0.00000 |
| mps | 32 | 924 | 8484 | 8.31906 | 501153792 | 0.00000 |
| mps | 64 | 528 | 4860 | 6.01777 | 559382528 | 0.00000 |

Independent device runs include the event loader in process peak RSS. Median score latency includes transfers and synchronized execution for a fixed validation sector. [Full run wall time and OS maximum RSS](process-time.txt)
