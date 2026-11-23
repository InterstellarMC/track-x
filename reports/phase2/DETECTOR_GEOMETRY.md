# Detector geometry and scope

The official geometry table contains 18728 modules. The reconstruction covers all azimuths in volume 8, divided into disjoint sectors. Module centres describe sensor placement; the minimum and maximum centre radii below do not describe complete sensor outlines.

| volume_id | layer_id | modules | r_min_mm | r_max_mm | z_min_mm | z_max_mm |
|---|---|---|---|---|---|---|
| 8 | 2 | 224 | 31.74997 | 32.24998 | -455.00000 | 455.00000 |
| 8 | 4 | 448 | 71.74999 | 72.25000 | -455.00000 | 455.00000 |
| 8 | 6 | 728 | 115.74964 | 116.25017 | -455.00000 | 455.00000 |
| 8 | 8 | 1092 | 171.74938 | 172.25043 | -455.00000 | 455.00000 |

TrackML has overlapping modules, unequal radial steps and disks at different z. A true particle may produce multiple hits on the same nominal layer. The Phase I generator permits at most one hit per layer. Radial ordering alone is unsuitable for endcaps. Geometry and occupancy for every volume are in [the detector figure](geometry.html) and [machine-readable inventory](geometry.json).
