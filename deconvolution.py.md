# Spatial Deconvolution with Cell2location
This workflow describes the process of cell-type deconvolution for Spatial Transcriptomics. Because a single Visium spot (55µm) usually contains multiple cells (typically 1–20), we cannot assume one spot equals one cell. This script uses Cell2location, a Bayesian model, to integrate single-cell RNA-seq (scRNA-seq) "signatures" into spatial data to estimate exactly which cell types are present in each spot.
## Background Info
Model Logic: How Bayesian deconvolution works.
Cell2location works in two distinct phases:
* **The Reference Model**: It looks at a single-cell atlas and learns what a "T-cell," "Fibroblast," or "Tumor Cell" looks like in terms of average gene expression ($mu$).
* **The Spatial Model**: It looks at the mixed signal in a Visium spot and calculates the combination of those signatures that best explains the observed counts, accounting for technical noise ($\alpha$).

## Steps
1. Reference Mapping: Establishing gene expression signatures for cell types.
2. Spatial Mapping: Projecting signatures onto tissue spots.
3. Visualization: Interpreting cell abundance maps.
