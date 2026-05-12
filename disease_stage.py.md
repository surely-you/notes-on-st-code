# Annotations on script.diease_stage.py
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
* **Differentially Expressed Genes (DEGs)**: genes whose expression levels are significantly upregulated or downregulated between conditions or groups
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
        # if a specific stage (like IPMN) isn't actually present in current dataset, it skips it to avoid errors.
        if stage not in adata.obs["disease_stage"].values:
            continue

        # creates a temporary column in metadata where every spot is labeled either "True" (it is the stage we are currently looking at) or "False" (it is any other stage).
        adata.obs["is_stage"] = (adata.obs["disease_stage"] == stage).astype(str)

        # U test
        sc.tl.rank_genes_groups(
            adata,
            groupby="is_stage", groups=["True"], reference="False",    # compares the "True" group against the "False" (reference) group.
            method="wilcoxon", use_raw=True,                           # uses the original, unlogged counts for the math, which is statistically more accurate for identifying DEGs.
        )

        # Extracting and Saving Results
        df = sc.get.rank_genes_groups_df(adata, group="True")
        df["stage"] = stage
        results[stage] = df
        df.to_csv(f"{OUTPUT_DIR}/DE_{stage}.csv", index=False)

        # Reporting Significance
        print(f"  {stage}: {(df['pvals_adj'] < 0.05).sum()} sig. DEGs")
    return results
```
## TME Composition Across Stages
This creates a Stacked Bar Chart showing the "recipe" of the tissue at each stage. It uses the deconvolution results to show how the proportions of T-cells, Fibroblasts, and Malignant cells shift over time.
* *comp = comp.div(comp.sum(axis=1), axis=0)*: Normalization
  * ensures each bar adds up to 100%, because different disease stages might have different total cell densities
  * It divides the value of each cell type by the total sum of all cell types in that stage.
  * converts raw numbers into percentages (0 to 1.0). 
```
# ── 2. TME composition across stages ─────────────────────────────────────────
def plot_tme_composition(adata: ad.AnnData):

    """Bar plots of mean cell-type abundance per disease stage."""

    # checks adata.obs to see which of your "Cell Types of Interest" actually exist in the data.
    # deconvolution didn't find any "Neural" cells, for example, it simply excludes that column so the code doesn't crash.
    avail_ct = [ct for ct in CELL_TYPES_OF_INTEREST if ct in adata.obs.columns]
    if not avail_ct:
        print("  No deconvolution columns found — skipping composition plot")
        return

    comp = (
        adata.obs.groupby("disease_stage")[avail_ct] # gathers all spots belonging to "Normal," then all spots belonging to "PanIN," and so on.
        .mean()
        .reindex(STAGE_ORDER) # ensures bars oare organized in biological order (Normal → PanIN → Primary → Met) rather than alphabetical order.
        .dropna(how="all")
    )

    # Normalize rows to sum to 1
    comp = comp.div(comp.sum(axis=1), axis=0)

    # Plotting the Stacked Bar Chart
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
Non-negative Matrix Factorization (NMF) to find "Gene Programs" (groups of genes that are co expressed), rather than looking at single genes.
### Non-negative Matrix Factorization (NMF)
**NMF:** technique used to break down large dataset into smaller meaningful parts while ensuring that all values remain non-negative. This helps in extracting useful features from data and making it easier to analyze and process it.
* decomposes a data matrix A into two smaller matrices W and H using an iterative optimization process that minimizes reconstruction error
  * W: Feature matrix (basis components)
  * H: Coefficient matrix (weights associated with W)
* source: https://www.geeksforgeeks.org/machine-learning/non-negative-matrix-factorization/

The application ($V \approx W \times H$) in our project:
* $V$ (The Input): original data (Spots $\times$ Genes).
* $W$ (The Spatial Map): Tells you where each program is active (Spots $\times$ Programs).
* $H$ (The Gene Dictionary): Tells you which genes belong to each program (Programs $\times$ Genes). For example, if Program 1 has high loadings for COL1A1 and FAP, it’s a Fibroblast/Stroma program.
```
# ── 3. NMF gene programs ──────────────────────────────────────────────────────
def extract_gene_programs(adata: ad.AnnData, n_programs: int = 10):
    """
    Run NMF on the log-normalized expression matrix (HVGs only) to extract
    latent gene programs. Saves top-50 genes per program.
    """

    # Filtering: NMF is computationally expensive. Instead of using all 20,000+ genes, the script only uses the Highly Variable Genes (HVGs) identified earlier.
    hvg_mask = adata.var.get("highly_variable", pd.Series(True, index=adata.var_names))
    X = adata[:, hvg_mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    # ensures there are absolutely no negative values in your matrix (set all - values to 0)
    X = np.clip(X, 0, None)   # NMF requires non-negative values

    # Factorization, see above description for W and H
    print(f"  Running NMF with {n_programs} components...")
    model = NMF(n_components=n_programs, init="nndsvda", random_state=42, max_iter=500)
    W = model.fit_transform(X)   # spots × programs
    H = model.components_        # programs × genes

    # Ranking and Saving the "Drivers"
    # looks at the H matrix, finds the top 50 genes for each program, and saves them to a CSV.
    gene_names = adata.var_names[hvg_mask]
    program_records = []
    for i, h in enumerate(H):
        top_idx  = np.argsort(h)[::-1][:50]
        top_genes = gene_names[top_idx].tolist()
        program_records.append({"program": f"P{i+1}", "top_genes": ", ".join(top_genes)})
    pd.DataFrame(program_records).to_csv(f"{OUTPUT_DIR}/nmf_programs.csv", index=False)

    # calculates the average activity score for each of the 10 programs across disease stages (Normal → PanIN → Primary → Met).
    for i in range(n_programs):
        adata.obs[f"NMF_P{i+1}"] = W[:, i]

    # Heatmap: mean program score per stage
    # Goal: To find programs that "ramp up" as the disease progresses. A "Malignancy Program" should be dark (low) in Normal tissue and bright yellow (high) in Metastasis.
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
create a single, comprehensive heatmap (specifically a matrixplot) that shows the most important gene markers for every stage of PDAC progression in one view. It picks the top 15 most significant genes from every stage and plots them in a grid.

```
# ── 4. Top DEG heatmap across stages ─────────────────────────────────────────
def plot_top_deg_heatmap(adata: ad.AnnData, de_results: dict, n_top: int = 15):
    top_genes = []
    for stage, df in de_results.items():
        sig = df[df["pvals_adj"] < 0.05].nlargest(n_top, "scores")    # filters for genes that are statistically significant and picks the "top 15" genes with the highest scores
        top_genes.extend(sig["names"].tolist())                       # collects all these genes into one master list.
    top_genes = list(dict.fromkeys(top_genes))                        # deduplicate, preserve order

    avail = [g for g in top_genes if g in adata.var_names]
    # Each row in the heatmap will be a Disease Stage, and each column will be a Gene.
    sc.pl.matrixplot(
        adata, var_names=avail,
        groupby="disease_stage", categories_order=STAGE_ORDER,  # forces the heatmap to display from top-to-bottom as Normal → PanIN → IPMN → Primary → Metastasis
        standard_scale="var",                                   # Z-scores everything (normaliza) to make relative gene expression comparable
        cmap="RdBu_r", save="_stage_deg_heatmap.png",           # Red-Blue color scheme: Red = high expression, blue = low expression
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
