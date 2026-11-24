# Fixed-path backward state estimation

The forward filter is frozen Phase IV M3. Each assigned outer hit supplies a predicted and updated state. The extended RTS gain is Pᵢ Fᵢᵀ(Pᵢ₊₁⁻)⁻¹ evaluated by Cholesky solves. Apply the gain to the wrapped smoothed-minus-predicted state and the covariance difference. No measurement or assignment changes. Compare seed-exit curvature estimates with the endpoint tangent-direction change across the pixel chord on pure extended validation tracks.

| event | count | filter_median_abs_error | smooth_median_abs_error | smooth_one_sigma | fit_s |
|---|---|---|---|---|---|
| event000001002 | 2774 | 0.011053 | 0.017350 | 0.750180 | 0.073838 |

The smoother worsened median absolute curvature error on this selected validation sample. It was excluded from reconstruction selection. The tighter forward gate is a separate development control. Material variation, displaced trajectories and the seed-to-endpoint proxy limit interpretation. Splitting was not evaluated.
