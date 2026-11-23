# Detector geometry and scope

The official geometry table contains 18728 modules. The reconstruction covers all azimuths in volume 8, divided into disjoint sectors. Module centres describe sensor placement; the minimum and maximum centre radii below do not describe complete sensor outlines.

| volume_id | layer_id | modules | r_min_mm | r_max_mm | z_min_mm | z_max_mm |
|---|---|---|---|---|---|---|
| 8 | 2 | 224 | 31.74997 | 32.24998 | -455.00000 | 455.00000 |
| 8 | 4 | 448 | 71.74999 | 72.25000 | -455.00000 | 455.00000 |
| 8 | 6 | 728 | 115.74964 | 116.25017 | -455.00000 | 455.00000 |
| 8 | 8 | 1092 | 171.74938 | 172.25043 | -455.00000 | 455.00000 |

Maximum elementwise rotation orthogonality error |RᵀR−I| was 1.27e-06; the determinant range was [0.999998734585, 1.000001246885]. Projecting hit-minus-centre onto the local axes gives normal distances and in-plane extent diagnostics. [Per-volume surface checks](surfaces.json)

| volume | hits | paths | efficiency | fake | official_score | median_abs_curvature_delta_per_m |
|---|---|---|---|---|---|---|
| 8 | 28339.66667 | 4883.33333 | 0.48630 | 0.48550 | 0.01552 | 0.00675 |
| 13 | 25430.66667 | 3479.33333 | 0.20366 | 0.76323 | 0.00043 | 0.00675 |
| 17 | 10214.33333 | 0.00000 | 0.00000 | undefined | 0.00000 | undefined |

The same single-pass geometric assembler was evaluated separately on each barrel volume. Volume 17 has only two nominal layers; this three-distinct-layer method cannot form candidates there. Curvature error compares fitted signed curvature with twice the sine of half the truth transverse momentum-direction change divided by truth transverse chord length. Both endpoints must belong to the matched particle. Material and field variation limit this local effective-curvature reference.

TrackML has overlapping modules, unequal radial steps and disks at different z. A true particle may produce multiple hits on the same nominal layer. The Phase I generator permits at most one hit per layer. Radial ordering alone is unsuitable for endcaps. Geometry and occupancy for every volume are in [the detector figure](geometry.html) and [machine-readable inventory](geometry.json).
