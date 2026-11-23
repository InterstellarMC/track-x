# Bounded measurement selection

Each active seed queries eight nearest measurements at most on each outer surface using the Phase III spatial index. The filter predicts again at the individual hit radius. A candidate must satisfy the turning-point check, |rΔφ|<35 mm, |Δz|<55 mm and innovation squared distance below 9. A deterministic sort by cost, seed and hit breaks ties. Local methods allow one hit per layer per seed and exclusive ownership. The fixed-window control substitutes Phase III geometric cost inside the same hard envelope.

M2 refreshes the robust fit and heuristic seed covariance after every accepted hit. M3 carries the propagated posterior forward. Both terminate on a missed layer. Absolute support, nearest-neighbor truncation, winner selection and gating censor the innovation distribution. Accepted innovations alone cannot test nominal two-degree-of-freedom coverage.

