# Annotated script.diease_stage.py
Now that the data is cleaned, deconvolved, and integrated, this script asks the biological questions: What genes change as the cancer progresses? and How does the neighborhood of cells evolve from a normal pancreas to a metastasis?
## Load libraries and set up environment 
```
"""
04_disease_stage_analysis.py
Stage-stratified differential expression and gene program analysis across:
  Normal → PanIN/IPMN → Primary PDAC → Metastasis

Outputs:
  - Per-stage DE gene tables
  - NMF gene programs per stage
  - TME cell-type composition summaries
  - Stage-transition heatmaps

Dependencies: scanpy, anndata, pandas, numpy, scipy, sklearn, matplotlib, seaborn
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import NMF
from scipy.stats import kruskal

OUTPUT_DIR = "results/disease_stage"
FIGURE_DIR = "figures/disease_stage"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

STAGE_ORDER = ["Normal_PDAC", "PanIN", "IPMN", "Primary_PDAC", "Metastasis"]

# Cell types to track from deconvolution (must match cell2location output columns)
CELL_TYPES_OF_INTEREST = [
    "Ductal (malignant)", "Ductal (normal)", "Acinar",
    "Stellate/CAF", "Endothelial", "Macrophage",
    "T cell", "B cell", "Neutrophil", "Neural",
]
```
## Per-Stage Differential Expression
This function identifies "Stage-Specific Markers." It compares every spot in one stage (e.g., PanIN) against all other spots in the atlas to find genes that are uniquely "turned on" during that phase of the disease.
### Algorithm: Wilcoxon Rank-Sum Test (aka Mann–Whitney U test)
statistical method for comparing two independent groups when your data doesn’t follow a normal (bell-shaped) distribution. 
* non-parametric version of the two-sample t-test
* it works by ranking all observations from both groups together and then checking whether one group’s ranks tend to be higher than the other’s
* source: https://scienceinsights.org/what-is-the-wilcoxon-rank-sum-test-and-when-to-use-it/ 

more details from COGS 108 slides (https://github.com/COGS108/Lectures-Sp26/blob/main/13-Nonparametric-Inference.pdf)
```
# ── 1. Per-stage differential expression ─────────────────────────────────────
def run_stage_de(adata: ad.AnnData) -> dict[str, pd.DataFrame]:
    """One-vs-rest DE for each disease stage."""
    results = {}
    for stage in STAGE_ORDER:
        if stage not in adata.obs["disease_stage"].values:
            continue
        adata.obs["is_stage"] = (adata.obs["disease_stage"] == stage).astype(str)
        sc.tl.rank_genes_groups(
            adata, groupby="is_stage", groups=["True"],
            reference="False", method="wilcoxon", use_raw=True,
        )
        df = sc.get.rank_genes_groups_df(adata, group="True")
        df["stage"] = stage
        results[stage] = df
        df.to_csv(f"{OUTPUT_DIR}/DE_{stage}.csv", index=False)
        print(f"  {stage}: {(df['pvals_adj'] < 0.05).sum()} sig. DEGs")
    return results
```
## TME Composition Across Stages
This creates a Stacked Bar Chart showing the "recipe" of the tissue at each stage. It uses the deconvolution results from Script 02 to show how the proportions of T-cells, Fibroblasts, and Malignant cells shift over time.
* *comp = comp.div(comp.sum(axis=1), axis=0)*: Normalization
  * ensures each bar adds up to 100%, because different disease stages might have different total cell densities
```
# ── 2. TME composition across stages ─────────────────────────────────────────
def plot_tme_composition(adata: ad.AnnData):
    """Bar plots of mean cell-type abundance per disease stage."""
    avail_ct = [ct for ct in CELL_TYPES_OF_INTEREST if ct in adata.obs.columns]
    if not avail_ct:
        print("  No deconvolution columns found — skipping composition plot")
        return

    comp = (
        adata.obs.groupby("disease_stage")[avail_ct]
        .mean()
        .reindex(STAGE_ORDER)
        .dropna(how="all")
    )
    # Normalize rows to sum to 1
    comp = comp.div(comp.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    comp.plot(kind="bar", stacked=True, colormap="tab20", ax=ax)
    ax.set_ylabel("Relative cell-type abundance")
    ax.set_xlabel("Disease stage")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/tme_composition_by_stage.png", dpi=150)
    plt.close()
    comp.to_csv(f"{OUTPUT_DIR}/tme_composition_by_stage.csv")
    print("  TME composition plot saved.")
```
## NMF Gene Programs
Non-negative Matrix Factorization (NMF) to find "Gene Programs" (groups of genes that always work together), rather than looking at single genes.
### Non-negative Matrix Factorization (NMF)
NMF is a "Parts-based Representation" algorithm. Imagine a spot is a "Sentence." NMF finds the "Words" (Gene Programs) that make up that sentence.
* **Matrix H:** Tells you which genes belong to which program (e.g., Program 1 = Inflammatory Genes).
* **Matrix W**: Tells you which spots are "using" that program.
* **Why it's "Non-negative"**: In biology, you can't have "negative" expression. NMF enforces that all values must be $\geq 0$, which results in "additive" programs that are much easier for biologists to interpret than the abstract components found in PCA.
```
# ── 3. NMF gene programs ──────────────────────────────────────────────────────
def extract_gene_programs(adata: ad.AnnData, n_programs: int = 10):
    """
    Run NMF on the log-normalized expression matrix (HVGs only) to extract
    latent gene programs. Saves top-50 genes per program.
    """
    hvg_mask = adata.var.get("highly_variable", pd.Series(True, index=adata.var_names))
    X = adata[:, hvg_mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.clip(X, 0, None)   # NMF requires non-negative values

    print(f"  Running NMF with {n_programs} components...")
    model = NMF(n_components=n_programs, init="nndsvda", random_state=42, max_iter=500)
    W = model.fit_transform(X)   # spots × programs
    H = model.components_        # programs × genes

    gene_names = adata.var_names[hvg_mask]
    program_records = []
    for i, h in enumerate(H):
        top_idx  = np.argsort(h)[::-1][:50]
        top_genes = gene_names[top_idx].tolist()
        program_records.append({"program": f"P{i+1}", "top_genes": ", ".join(top_genes)})

    pd.DataFrame(program_records).to_csv(f"{OUTPUT_DIR}/nmf_programs.csv", index=False)

    # Store program scores on obs
    for i in range(n_programs):
        adata.obs[f"NMF_P{i+1}"] = W[:, i]

    # Heatmap: mean program score per stage
    prog_cols = [f"NMF_P{i+1}" for i in range(n_programs)]
    prog_stage = adata.obs.groupby("disease_stage")[prog_cols].mean().reindex(STAGE_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(prog_stage.T, cmap="viridis", ax=ax, yticklabels=prog_cols)
    ax.set_title("NMF gene program activity across disease stages")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/nmf_program_heatmap.png", dpi=150)
    plt.close()
    print(f"  NMF programs saved.")
    return adata
```
## Top DEG Heatmap
This creates a visual "Master Summary" of the atlas. It picks the top 15 most significant genes from every stage and plots them in a grid.
```
# ── 4. Top DEG heatmap across stages ─────────────────────────────────────────
def plot_top_deg_heatmap(adata: ad.AnnData, de_results: dict, n_top: int = 15):
    top_genes = []
    for stage, df in de_results.items():
        sig = df[df["pvals_adj"] < 0.05].nlargest(n_top, "scores")
        top_genes.extend(sig["names"].tolist())
    top_genes = list(dict.fromkeys(top_genes))   # deduplicate, preserve order

    avail = [g for g in top_genes if g in adata.var_names]
    sc.pl.matrixplot(
        adata, var_names=avail, groupby="disease_stage",
        categories_order=STAGE_ORDER, standard_scale="var",
        cmap="RdBu_r", save="_stage_deg_heatmap.png",
    )
    print("  DEG heatmap saved.")
```
## Main
calls the aforementioned functions and ties everything together
```
# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading integrated atlas...")
    adata = sc.read_h5ad("data/integrated/pdac_atlas_integrated.h5ad")

    print("Running per-stage differential expression...")
    de_results = run_stage_de(adata)

    print("Plotting TME composition by stage...")
    plot_tme_composition(adata)

    print("Extracting NMF gene programs...")
    adata = extract_gene_programs(adata, n_programs=10)

    print("Plotting top DEG heatmap across stages...")
    plot_top_deg_heatmap(adata, de_results)

    adata.write_h5ad("data/integrated/pdac_atlas_with_programs.h5ad")
    print("\nDone. Results in results/disease_stage/ and figures/disease_stage/")
```
