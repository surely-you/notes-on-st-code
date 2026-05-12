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


# ── Load all deconvolved samples ──────────────────────────────────────────────
def load_all_samples(deconv_dir: str) -> ad.AnnData:
    adatas = []
    for sid, stage in SAMPLE_STAGES.items():
        p = f"{deconv_dir}/{sid}_deconvolved.h5ad"
        if not os.path.exists(p):
            p = f"data/processed/{sid}_processed.h5ad"  # fallback
        if not os.path.exists(p):
            print(f"  Warning: {sid} not found, skipping")
            continue
        a = sc.read_h5ad(p)
        a.obs["sample_id"]     = sid
        a.obs["disease_stage"] = stage
        adatas.append(a)
        print(f"  Loaded {sid} ({stage}): {a.n_obs} spots")

    # Concatenate on shared genes
    combined = ad.concat(adatas, join="inner", label="sample_id",
                          keys=[a.obs["sample_id"].iloc[0] for a in adatas])
    combined.obs_names_make_unique()
    print(f"\nCombined: {combined.n_obs} spots, {combined.n_vars} genes")
    return combined


# ── Re-normalize after concatenation ─────────────────────────────────────────
def renormalize(adata: ad.AnnData) -> ad.AnnData:
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=3000, flavor="seurat_v3",
        layer="counts", batch_key="sample_id"
    )
    return adata


# ── Integration: Harmony ──────────────────────────────────────────────────────
def integrate_harmony(adata: ad.AnnData) -> ad.AnnData:
    import harmonypy as hm
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, ["sample_id"])
    adata.obsm["X_integrated"] = ho.Z_corr.T
    sc.pp.neighbors(adata, use_rep="X_integrated", n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    return adata


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
