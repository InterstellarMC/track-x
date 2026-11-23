# Phase III replay

On the local Apple M1, the locked Python 3.12 environment passed all 31 original tests. The 12 warnings came from pandas operations in the reference scorer. The isolated Phase II replay matched the archived scientific results for all three test events, manifest and training metadata. The Phase III replay matched every non-timing method field and supported weight on 1009–1011 using revision 0f1e0cc. The checker is scripts/audit_phase4.py.

| Replay | Official full-event score | Efficiency | Mixed | Reconstruction seconds/event |
|---|---:|---:|---:|---:|
| Two-pass pixel | .01858332 | .030715 | .335123 | 1.030 |
| Extended geometry | .22319501 | .266443 | .416957 | 1.270 |
| GNN-ranked extended | .22307703 | .266437 | .417060 | 1.518 |
| No-gap | .22393086 | .268354 | .399263 | 1.168 |

Runtime varies between invocations. Historical events and the original weights remain in their existing locations. The new protocol was committed at 993b8e1 before fresh data acquisition.

