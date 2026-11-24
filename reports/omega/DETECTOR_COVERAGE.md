# Forward pixel disk feasibility

Inspect absolute median module z by volume and layer. Volumes 7 and 9 have seven signed-z surfaces each. For the first two planes use a prompt-origin radial scaling and at most four neighbors within 12 mm in xy. A linear xy extrapolation to the third median plane accepts the nearest candidate within 2 mm. Claim disjoint triplets by residual, then extend with a 3 mm gate on subsequent disks. Terminate on misses. This exploratory follower omits curvature transport, barrel-to-disk linking and the outer strip disks.

| volume | ordered (absolute z in metres, layer) |
|---|---|
| 7 | [[0.6, 14], [0.7, 12], [0.82, 10], [0.96, 8], [1.1, 6], [1.3, 4], [1.5, 2]] |
| 9 | [[0.6, 2], [0.7, 4], [0.82, 6], [0.96, 8], [1.1, 10], [1.3, 12], [1.5, 14]] |

| development event | barrel score | disk only | disjoint union | disk seconds | vol 7 mixed | vol 9 mixed |
|---|---|---|---|---|---|---|
| event000001003 | 0.232477 | 0.060930 | 0.293408 | 0.072550 | 0.094237 | 0.109899 |
| event000001004 | 0.213652 | 0.053950 | 0.267602 | 0.088308 | 0.117916 | 0.136842 |
| event000001005 | 0.225136 | 0.056647 | 0.281782 | 0.065880 | 0.114884 | 0.100656 |

All three scores use complete-event truth. These exploratory arms were excluded from Omega confirmation. Per-event truth weights for all nine volumes accompany the main benchmark.
