# Full-event scoring

| event | baseline | extended | paired_delta |
|---|---|---|---|
| event000001009 | 0.01891 | 0.22645 | 0.20753 |
| event000001010 | 0.01721 | 0.21926 | 0.20205 |
| event000001011 | 0.01962 | 0.22388 | 0.20426 |

The pinned organizers’ score_event receives every hit, with a single shared track ID zero for unassigned hits. The score uses strict majority in both directions and correctly assigned normalized truth weight. Custom track matching uses purity >0.5 and completeness ≥0.5 on all observed event hits, including disks. Truth particles require at least three observed hits. Fake and duplicate rates divide by output tracks. Purity and completeness average first qualifying matches. Mixed fraction counts paths containing hits of two or more nonzero particle IDs; it also counts one-hit contaminants. Coverage is owned hits / all event hits.

| event | supported_weight |
|---|---|
| event000001002 | 0.58310 |
| event000001003 | 0.57358 |
| event000001004 | 0.57136 |
| event000001005 | 0.57496 |
| event000001009 | 0.57092 |
| event000001010 | 0.57530 |
| event000001011 | 0.57554 |

Supported weight sums original truth weights in volumes 8, 13 and 17 and divides by full-event truth weight. It measures available weight before reconstruction. Regional matched-hit efficiency divides correctly assigned hits from full-event matched paths by all signal hits in that region.

| event | 8 | 13 | 17 |
|---|---|---|---|
| event000001009 | 0.36427 | 0.41192 | 0.48129 |
| event000001010 | 0.35149 | 0.39669 | 0.46569 |
| event000001011 | 0.35250 | 0.40066 | 0.46583 |
