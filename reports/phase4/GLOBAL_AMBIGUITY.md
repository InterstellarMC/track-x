# Whole-path conflict resolution

M4 uses fixed geometric cost, M5 uses EKF innovation cost. Each seed independently forms its minimum-cost continuation, then proposes every nonempty prefix. Each accepted outer hit contributes 1−cost/9. Candidates compete by descending total compatibility with deterministic ties. An occupied seed or occupied hit removes that candidate from consideration. Each unextended seed is retained. The search is bounded by six outer layers and eight neighbors per layer. The objective rewards additional compatible hits and gives marginal hits little credit. Missing a layer terminates a proposal. Greedy packing can miss a higher-total combination and the prefix pool omits alternate branches. The M3–M5 comparison measures replacing layer-by-layer assignment with this specific whole-path procedure.

The recorded accepted sample is before whole-path selection; final_accepted shows ownership after resolution. altered_by_conflict counts seeds whose independently proposed path changed.

