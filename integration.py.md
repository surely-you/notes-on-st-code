# Annotations on script_integration.py
This script represents the final "assembly" of your data. While the previous scripts handled individual samples and deconvolution, 03_integration.py merges all samples into a single Unified PDAC Atlas.

The goal here is to remove batch effects—technical differences caused by different patients, labs, or sequencing runs—so that biological differences (like Normal vs. Cancer) can be studied clearly.


## Load libraries and set up environment 
### Harmony (Faster)
* linear integration method. It assumes that if you have two different datasets, the "biological clusters" (like T-cells) are actually the same shape, just shifted to different spots in space due to technical noise.
* How it works:
  * **PCA First**: Harmony starts with your standard PCA coordinates.
  * **Clustering**: It groups all spots into "soft clusters."
  * **The Correction**: For each cluster, it calculates a "center" for each batch (e.g., the center of Sample A's T-cells vs. Sample B's T-cells).
  * **Moving the Spots**: It acts like a magnet, pulling the spots from different batches toward a shared center for that specific cluster.
  * **Iteration**: It repeats this until the batches are mixed together, but the biological clusters remain distinct.
* **Key Strength**: It is incredibly fast and preserves the original "shape" of your data. It doesn't change the gene counts; it only changes the coordinates used for plotting and clustering.
COGS 108: PCA what and how (https://github.com/COGS108/Lectures-Sp26/blob/main/16-Dimensionality-Reduction.pdf)

### scVI (more powerful)
* scVI (single-cell Variational Inference) is a non-linear method based on Deep Learning. It doesn't just move points around; it tries to rebuild your data from scratch.

* **How it works**: It uses a Neural Network to model the "raw counts" of the data. It assumes that every count is a mixture of Biology and Batch.
  * **The Encoder**: A Neural Network looks at your raw gene counts and "compresses" them into a small set of numbers (the Latent Space).
  * **The Bottleneck**: During this compression, the model is forced to separate "Biology" from "Batch."
    * It is told: "Here is the Count, and here is the Batch ID. Try to explain the Count using as few biological variables as possible."
  * **The Decoder**: Another Neural Network tries to take those compressed biological variables and reconstruct the original counts.
  * **The Result**: The "Latent Space" (X_scVI) is a set of coordinates where the batch effect has been mathematically filtered out by the neural network.
* **Why use it for PDAC?** Since your data comes from many different sources (GEO, 10x, etc.), scVI is better at handling the "complex noise" found across different technological platforms.
```
"""
03_integration.py
Integrate all PDAC spatial transcriptomics datasets into a unified atlas.
Uses Harmony for PCA-level batch correction (fast) or scVI for count-level
integration (more powerful across platforms). Both options are provided.

Dependencies: scanpy, anndata, harmonypy, scvi-tools, pandas, numpy, matplotlib
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Optional: uncomment the integration method you want
# METHOD = "harmony"
METHOD = "scvi"

OUTPUT_DIR = "data/integrated"
FIGURE_DIR = "figures/integration"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

STAGE_ORDER = ["Normal", "PanIN", "IPMN", "Primary_PDAC", "Metastasis"]

SAMPLE_STAGES = {
    "GSE254829"  : "PanIN",
    "GSE233293"  : "IPMN",
    "GSE327056"  : "Normal_PDAC",
    "GSE274103"  : "Primary_PDAC",
    "GSE310353"  : "Primary_PDAC",
    "Syn61831984": "Primary_PDAC",
    "GSE278694"  : "Primary_PDAC",
    "GSE272362"  : "Metastasis",
    "GSE274557"  : "Metastasis",
    "10x_VisiumHD": "Primary_PDAC",
}
```
## Data Concatenation
merges the individual sample files into a single master dataset.
* *combined = ad.concat(adatas, join="inner", label="sample_id")*: When you merge multiple AnnData objects, you create a massive matrix. This script uses an inner join, meaning it only keeps genes that were detected in every sample across your manifest.
  * *label="sample_id"*: This automatically creates a new metadata column to keep the source samples distinct.
```
# ── Load all deconvolved samples ──────────────────────────────────────────────
def load_all_samples(deconv_dir: str) -> ad.AnnData:
    adatas = []
    for sid, stage in SAMPLE_STAGES.items():
        # looks for the deconvolved version of the sample (the one that contains the cell-type counts from the previous step)
        p = f"{deconv_dir}/{sid}_deconvolved.h5ad"
        # If deconvolution failed or wasn't run for a specific sample, it falls back to the processed version (just the gene counts)
        if not os.path.exists(p):
            p = f"data/processed/{sid}_processed.h5ad"  # fallback
        if not os.path.exists(p):
            print(f"  Warning: {sid} not found, skipping")
            continue

        # tags every single spot in that sample to keep track of where the samples came from and cancer stage
        a = sc.read_h5ad(p)
        a.obs["sample_id"]     = sid
        a.obs["disease_stage"] = stage
        adatas.append(a)
        print(f"  Loaded {sid} ({stage}): {a.n_obs} spots")

    # Concatenate on shared genes
    combined = ad.concat(adatas, join="inner", label="sample_id",
                          keys=[a.obs["sample_id"].iloc[0] for a in adatas])

    # spatial samples use the same barcode naming system might end up with two different spots with the same name
    # This function appends a suffix to the spot names so that every row in the new master matrix has a unique ID.
    combined.obs_names_make_unique()
    print(f"\nCombined: {combined.n_obs} spots, {combined.n_vars} genes")
    return combined
```
## Sanity Check and Reset
normalize and prepares the merged dataset for Integration
* *sc.pp.normalize_total(adata, target_sum=1e4)*: One Visium sample might have more total reads than another simply because the sequencer ran longer, not because the biology is different.
  * Adjusts each spot so that all counts add up to 10,000. This makes the "volume" of expression comparable across all 30,000+ spots in your atlas.
  * *log1p*: $log(1+x)$ transformation.
* *sc.pp.highly_variable_genes*: Instead of just picking the 3,000 most variable genes across the entire merged dataset, it does so per sample.
  * Result: You get a list of 3,000 genes that truly represent the biological differences in your PDAC progression (e.g., genes that distinguish a tumor cell from a fibroblast, regardless of which patient they came from).
  * *batch_key="sample_id"*: This tells the model to find genes that are variable within each sample, and then select the ones that are consistently variable across all samples.
```
# ── Re-normalize after concatenation ─────────────────────────────────────────
def renormalize(adata: ad.AnnData) -> ad.AnnData:
    # Preserving the Raw "Counts"
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Global Normalization & Scaling
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Batch-Aware Feature Selection (HVGs)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=3000, flavor="seurat_v3",
        layer="counts", batch_key="sample_id" # tells the model to find genes that are variable within each sample, and then select the ones that are consistently variable across all samples.
    )
    return adata
```
## Merging the datasets using harmony
* *sc.tl.pca(adata, n_comps=50, use_highly_variable=True)* : Compresses the 3,000 Highly Variable Genes into 50 "Principal Components"
  * Harmony doesn't work on raw genes; it works on the mathematical summaries (PCs) of those genes. This removes noise and speeds up the calculation.
```
# ── Integration: Harmony ──────────────────────────────────────────────────────
def integrate_harmony(adata: ad.AnnData) -> ad.AnnData:

    import harmonypy as hm
    # z score standardization (mean = 0, variance = 1)
    sc.pp.scale(adata, max_value=10)

    # Compresses the 3,000 Highly Variable Genes into 50 "Principal Components"
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

    # runs harmony
    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, ["sample_id"])
    adata.obsm["X_integrated"] = ho.Z_corr.T

    # Neighbors, UMAP, and Clustering
    sc.pp.neighbors(adata, use_rep="X_integrated", n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    return adata
```
## merging the datasets using scVI
*model = scvi.model.SCVI(adata_scvi, n_layers=2, n_latent=30, gene_likelihood="nb")*: Because scVI is a generative model, it has to "predict" what the gene counts should look like. By telling it to use a Negative Binomial distribution, you are telling the neural network: "Expect the data to be noisy and overdispersed." This prevents the AI from over-fitting to technical glitches and helps it focus on the true biological signal.

```
# ── Integration: scVI ─────────────────────────────────────────────────────────
def integrate_scvi(adata: ad.AnnData) -> ad.AnnData:
    import scvi
    scvi.settings.seed = 42
    adata_scvi = adata[:, adata.var["highly_variable"]].copy()
    adata_scvi.X = adata_scvi.layers["counts"]   # scVI needs raw counts

    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        layer="counts",
        batch_key="sample_id",
    )
    model = scvi.model.SCVI(adata_scvi, n_layers=2, n_latent=30, gene_likelihood="nb")
    model.train(max_epochs=400, early_stopping=True)
    model.save("models/scvi_integration", overwrite=True)

    adata.obsm["X_scVI"] = model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    return adata
```
## annotate clusters (consider SingleR?)
```
# ── Cluster annotation helpers ────────────────────────────────────────────────
MARKER_GENES = {
    "Ductal (malignant)" : ["KRT19", "EPCAM", "KRT8", "MUC1"],
    "Ductal (normal)"    : ["KRT18", "CFTR", "SLC4A4"],
    "Acinar"             : ["PRSS1", "CELA1", "REG1A"],
    "Stellate/CAF"       : ["ACTA2", "FAP", "POSTN", "COL1A1"],
    "Endothelial"        : ["PECAM1", "VWF", "CDH5"],
    "Macrophage"         : ["CD68", "CD163", "MRC1"],
    "T cell"             : ["CD3D", "CD8A", "CD4"],
    "B cell"             : ["CD19", "MS4A1"],
    "NK cell"            : ["NCAM1", "GNLY"],
    "Neutrophil"         : ["S100A8", "S100A9", "FCGR3B"],
    "Neural"             : ["S100B", "PLP1", "MPZ"],    # for perineural invasion
}

def annotate_clusters(adata: ad.AnnData) -> ad.AnnData:
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True)
    sc.pl.dotplot(adata, var_names=MARKER_GENES, groupby="leiden",
                  save="_cluster_markers.png")
    return adata
```
## Main
ties together all the aforementioned functions and calls everything 
```
# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading all samples...")
    adata = load_all_samples("data/deconvolved")

    print("Renormalizing concatenated object...")
    adata = renormalize(adata)

    print(f"Integrating with {METHOD}...")
    if METHOD == "harmony":
        adata = integrate_harmony(adata)
    else:
        adata = integrate_scvi(adata)

    print("Annotating clusters...")
    adata = annotate_clusters(adata)

    # Color UMAPs by key covariates
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, color in zip(axes, ["disease_stage", "sample_id", "leiden"]):
        sc.pl.umap(adata, color=color, ax=ax, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/atlas_umap.png", dpi=150)
    plt.close()

    out = f"{OUTPUT_DIR}/pdac_atlas_integrated.h5ad"
    adata.write_h5ad(out)
    print(f"\nAtlas saved → {out}")
```
