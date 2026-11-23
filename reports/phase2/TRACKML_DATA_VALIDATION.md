# Official sample validation

The authenticated acquisition uses range reads of six complete event prefixes in sorted order. CRC checks on extracted members and SHA-256 identities freeze inputs. Every event passed finite numeric, unique key, hit/truth alignment, particle reference, detector module and cell association checks. Particle IDs retain integer precision. The official source describes simulated silicon tracking measurements. [Organizers](https://sites.google.com/site/trackmlparticle/dataset) · [Frozen manifest](manifest.json)

| event | hits | cells | particles | weight_sum | volume8_hits | noise_hits | barrel_particles | repeated_layer_particles | particles_lt3_distinct_layers |
|---|---|---|---|---|---|---|---|---|---|
| event000001000 | 120939 | 664996 | 12263 | 1.00000 | 29172 | 2217 | 7185 | 3701 | 2334 |
| event000001001 | 93680 | 516424 | 8915 | 1.00000 | 21941 | 2171 | 5296 | 2749 | 1805 |
| event000001002 | 125504 | 689199 | 12763 | 1.00000 | 30459 | 2173 | 7576 | 3876 | 2548 |
| event000001003 | 104451 | 575561 | 10261 | 1.00000 | 24979 | 2179 | 6056 | 3101 | 1934 |
| event000001004 | 138568 | 761134 | 14159 | 1.00000 | 33799 | 2221 | 8370 | 4383 | 2759 |
| event000001005 | 110023 | 608716 | 10945 | 1.00000 | 26241 | 2151 | 6482 | 3267 | 2199 |

Input sizes, full split membership and hashes are in manifest.json. The cells table is streamed in chunks; it is validated but supplies no inference features.
