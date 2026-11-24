# Release and publication readiness

This local research package is suitable for technical review and reproduction. It contains a prospective protocol, historical controls, complete-event results, raw-file hashes, method sources, test fixtures, machine-readable diagnostics, printable figures and a manuscript. Previous phases and their released numbers remain available. The MIT license covers project code. Obtain simulated events through the TrackML competition under its own access terms. Include the Git revision, configuration hash and input manifest with software citations.

A publication decision requires external review. Principal limitations are event selection within one simulation production, approximate geometry and covariance, a limited set of competing algorithms and incomplete detector coverage. A reference ACTS comparison is open. Disk and smoothing studies are exploratory. Before external release, verify attribution, authorship and dataset redistribution terms with the project owner. No raw event CSVs or credentials belong in a release.

The Markdown manuscript is the source. Generate the typesetting sources with scripts/omega_manuscript.py after building figures. In reports/omega run `tectonic manuscript.tex` for a LaTeX PDF on a system with its TeX bundle available. The printable HTML uses the same Markdown and an offline vector figure. Reviewer access needs the locally generated files or an explicitly authorized public release.

The three-page manuscript PDF was rendered and visually inspected. The LaTeX source compiled with Tectonic after its font dependency was cached, and a second offline build completed successfully. Local release verification is recorded in [the clean-checkout receipt](clean-replay.json).
