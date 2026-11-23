# TRACK-X | Prompt barrel-track reconstruction on an Apple M1

## Abstract

We compared four reconstruction pipelines on held-out synthetic events from an eight-layer barrel. A compact edge GNN recovered 75.0% of eligible tracks in a combined stress sample, compared with 59.9% for the edge MLP and 71.4% for a geometric follower. Adding a triplet constraint reduced fake tracks from 9.2% to 6.6% with a small efficiency loss. Nominal events approached saturation even for the geometric baseline. These measurements characterize the simulator, reconstruction choices and test protocol described below.

## Problem and physical model

Measurements arrive without trajectory identities. A single outward charged particle provides position constraints at successive radii. In a uniform solenoid its transverse path is a signed circle and its longitudinal coordinate is linear in transverse arc length. Increasing occupancy creates competing links; missing measurements create gaps. Noise hits can bridge two unrelated trajectories. Recovering a coherent sequence therefore requires decisions beyond local edge classification. Model equations and units are in [Physics assumptions](../docs/PHYSICS_ASSUMPTIONS.md).

## Related work

The TrackML challenge supplies simulated detector hits, cell measurements, particles and truth associations, with a weighted challenge score. Its reference implementation distinguishes hit identifiers from 64-bit particle identifiers. TRACK-X uses a local overlap metric and a separate ideal geometry. [TrackML accuracy phase](https://arxiv.org/abs/1904.06778) · [Reference loader](https://github.com/LAL/trackml-library/blob/master/trackml/dataset.py)

Graph-based tracking research includes learned embeddings for track seeding and edge classification. Here the candidate builder uses fixed geometric windows so graph acceptance can be audited independently of learned scores. ACTS describes surface-based track parameterization, propagation, covariance transport and treatment of material. Those capabilities define a substantial validation gap for this small barrel model. [Graph seeding and labelling](https://arxiv.org/abs/2007.00149) · [ACTS tracking concepts](https://acts.readthedocs.io/en/latest/tracking.html)

## Data and methods

Each of three seeds supplies 60 nominal training events, ten validation events and 15 held-out test IDs per condition. The stress condition changes occupancy, clutter, vertex spread, inefficiency and the pT distribution together. The pileup proxy samples five shared beam-axis vertices per event. Physically separated truth arrays are used for supervised targets and post-reconstruction evaluation. No official TrackML event was used in the results.

A periodic spatial search proposes directed edges over one or two layer transitions. Each classifier uses endpoint measurements and relative geometry. The MLP has no message updates. The GNN uses two steps with hidden dimension 32. Class-weighted BCE addresses imbalance; resulting sigmoid scores have no probability calibration guarantee. All methods use a fixed 0.5 operating point. Exclusive greedy assembly prioritizes high-score links. Geometric and hybrid methods also enforce smooth local directions. Fit quality is evaluated after assembly. [Architecture](../docs/ARCHITECTURE.md)

## Evaluation and results

Track matching requires strictly majority purity and at least half completeness. Comparison uses identical hits and graph edges for all four methods within an event. Macro-averaging weights events equally, then weights seeds equally. [Exact definitions](../docs/EXPERIMENT_PROTOCOL.md) · [Generated tables](RESULTS.md) · [Interactive comparison](efficiency.html)

- **Nominal.** Classical efficiency was 99.9% and GNN efficiency was 99.7%. Both leave little room for improvement under the selected matching rule. Candidate-edge AP was 0.981 and 0.993 respectively. Edge ranking and track recovery measure different outcomes.
- **Combined stress.** GNN efficiency was 0.750 ± 0.020 versus 0.599 ± 0.011 for the MLP. The GNN introduces both message exchange and additional parameters. This comparison measures their combined effect. The classical follower remained competitive at 0.714 ± 0.035.
- **Constraint ablation.** Adding the triplet test to identical GNN scores yielded efficiency 0.746 ± 0.015 and fake rate 0.066 versus 0.750 ± 0.020 and 0.092 without the gate. The edge AP stays identical by construction. The experiment supports an operating-point trade-off. Three seeds give limited precision for small differences.
- **Shared vertices.** Mean efficiencies were 0.769 classical, 0.644 MLP, 0.826 GNN and 0.826 hybrid. Hybrid fake rate was 0.078 versus 0.088 for GNN. Multiple detector crossings, realistic luminous-region dynamics and bunch timing are absent.
- **Graph acceptance.** A separate truth audit recovered every same-particle hit pair separated by one or two layers in these test samples. Longer gaps are excluded from that denominator. This highlights the simplicity of the synthetic geometry. [Audit counts](graph-acceptance.json)
- **Fit study.** Across 479 matched classical candidates on nominal seed 1729, median absolute signed-curvature error was 0.00784 m⁻¹ for the algebraic fit and 0.00735 m⁻¹ for the robust nonlinear fit. Both used identical hit sequences. This conditional test covers reconstructed matches only. [Fit residual records](fit-study.json)

## Computing and reproducibility

The smoke experiment and mac_m1 training ran natively on the 8 GiB Apple M1. A fixed-checkpoint microbenchmark on four smoke graphs gave median GNN scoring times of 0.41 ms on CPU and 2.39 ms on MPS, including transfer back to CPU. The benchmark uses 40 timed calls per device. CPU is a suitable default at this graph size. Total reconstruction timing additionally includes Python path assembly and trajectory fitting. Hardware, commands and peak process measurements are in the [manifest](MANIFEST.md).

## Limitations and future work

The geometry prior assumes prompt outward motion. Sagitta becomes poorly resolved at high momentum. The field model lacks material interactions, segmentation, alignment errors and endcaps. The path builder can lock in an incorrect hit early. The acceptance audit, high nominal efficiencies and permissive half-completeness threshold all limit comparison with detector reconstruction. There is no test of full-event high-occupancy scaling, pull calibration or transfer to collision data. Extended configuration benchmark and official TrackML metrics: **NOT RUN**.

The highest-value follow-up is a frozen TrackML barrel-sector study. Measure true-edge acceptance, matched-track efficiency, fake rate and parameter residuals against pT and occupancy. Split official files by event and compare synthetic-transfer weights with a model trained on official simulated events. Document the field convention and module geometry before interpreting fitted momenta. [Upstream roadmap](../docs/ROADMAP.md)
