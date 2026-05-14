# Annotations on script_deconvolution.py
combines Transcriptomic Similarity (Ligand-Receptor pairs) with Spatial Proximity (Co-occurrence) to see how different cell types communicate as cancer progresses.

## Summary
1. **⭐LIANA (Ligand-Receptor Analysis)** 
2. filter & plot results
3. **⭐Squidpy Spatial Co-occurrence**
4. Visualization & Comparison

## Setting up
new libraries:
* **liana**: run several different methods for Cell-Cell Communication (CCC) analysis simultaneously and provides a consensus score.
```
"""
05_cell_cell_interactions.py
Cell-cell interaction analysis using LIANA (Python) and spatial co-occurrence
via Squidpy. Identifies ligand-receptor pairs at the tumor-stroma interface
and compares communication patterns across disease stages.

Dependencies: liana, squidpy, scanpy, anndata, pandas, numpy, matplotlib
"""

import scanpy as sc
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import liana
import os

OUTPUT_DIR = "results/cci"
FIGURE_DIR = "figures/cci"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

STAGE_ORDER    = ["Normal_PDAC", "PanIN", "IPMN", "Primary_PDAC", "Metastasis"]
CELL_TYPE_COL  = "cell_type_annotation"   # set in 03_integration.py
KEY_PAIRS = [                              # sender → receiver pairs of interest
    ("Ductal (malignant)", "Stellate/CAF"),
    ("Ductal (malignant)", "Macrophage"),
    ("Stellate/CAF",       "T cell"),
    ("Macrophage",         "Ductal (malignant)"),
    ("Neural",             "Ductal (malignant)"),   # perineural invasion axis
]
```
## 1. LIANA (Ligand-Receptor Analysis)

Core Functions
* **Purpose**: Identifies potential "conversations" between cells.
* **Logic**: It looks for pairs where Cell A expresses a Ligand (signal) and Cell B expresses the matching Receptor.
* **Method Aggregation**: implements multiple scoring functions (e.g., expression product, truncated mean, etc.) to ensure the results aren't biased by one specific algorithm's quirks.
* **Resource Management:** provides access to a massive database of Ligand-Receptor (LR) pairs, including protein complexes (where a signal is made of multiple subunits).
* **Ranking**: "aggregate rank" system. If multiple methods agree that an interaction is important, it gets a high rank.
* **Result**: A table of sender $\rightarrow$ receiver pairs, ranked by how likely they are to be interacting.
<img width="2336" height="1334" alt="image" src="https://github.com/user-attachments/assets/98c9ac46-f92a-4d64-a3fb-a2dbd6e27c2d" />
source: https://github.com/saezlab/liana

-----

$\downarrow$ helper function that runs LIANA for a single stage 

```
# ── 1. LIANA ligand-receptor analysis ────────────────────────────────────────
def run_liana(adata: ad.AnnData, stage: str) -> pd.DataFrame:

    """Run LIANA consensus LR scoring for one disease stage subset."""

    # takes an AnnData object (the standard format for single-cell data) and slices it to include only the cells or spots that belong to a specific stage
    sub = adata[adata.obs["disease_stage"] == stage].copy()

    # Statistical methods for ligand-receptor interactions require a minimum number of cells to be reliable. If the subset has fewer than 50 cells/spots, the function prints a warning and returns an empty table. This prevents the code from crashing or producing "noisy" results based on insufficient data.
    if sub.n_obs < 50:
        print(f"  {stage}: too few spots, skipping")
        return pd.DataFrame()

    # Instead of relying on one method, LIANA runs several (like CellPhoneDB, NATMI, etc.) and calculates a consensus rank to makes the results more robust.
    liana.mt.rank_aggregate(
        sub,
        groupby=CELL_TYPE_COL,   # looks for interactions between the categories defined in your cell type column
        use_raw=True,            # uses the raw gene expression counts
        verbose=False,
    )

    # stores its output in the .uns (unstructured) dictionary of the AnnData object under the key "liana_res".
    res = sub.uns["liana_res"].copy()
    # extracts that table and adds a new column called "stage"
    res["stage"] = stage
    return res
```
calls helper function in a for loop that runs LIANA for all stages 
```
def run_liana_all_stages(adata: ad.AnnData) -> pd.DataFrame:

    # empty list to store the results from each individual stage as they are processed.
    all_res = []

    # for loop that automagically does everything
    for stage in STAGE_ORDER:
        print(f"  Running LIANA for {stage}...")
        df = run_liana(adata, stage)
        all_res.append(df)

    # takes that list of individual DataFrames and stacks them on top of each other into one giant master table.
    combined = pd.concat(all_res, ignore_index=True)

    #writed output to csv
    combined.to_csv(f"{OUTPUT_DIR}/liana_all_stages.csv", index=False)
    return combined
```
## 2. filter and plot top relevant ones
narrows down potential interactions to the most biologically relevant across all disease stages. (the filter step)
* *aggregate_rank*: similar to p value 
```
# ── 2. Filter & rank interactions ─────────────────────────────────────────────
def get_top_interactions(liana_res: pd.DataFrame, n_top: int = 20) -> pd.DataFrame:
    """Filter to significant interactions and rank by aggregate rank score."""
    sig = liana_res[liana_res["aggregate_rank"] < 0.05].copy() #

    # Since the data has multiple stages of cancer, an interaction might appear multiple times.
    ranked = (
        sig.groupby(["source", "target", "ligand_complex", "receptor_complex"])
        .agg(mean_rank=("aggregate_rank", "mean"), # Calculates the average strength of that interaction across all stages where it was significant.
            n_stages=("stage", "nunique"))         # Counts how many different disease stages this specific interaction appeared in (its "persistence")
        .reset_index()                             # sorts the results so that the interactions with the lowest mean rank (the strongest, most consistent ones) are at the top of the list
        .sort_values("mean_rank")
    )

    ranked.to_csv(f"{OUTPUT_DIR}/top_interactions_consensus.csv", index=False)
    return ranked.head(n_top)
```
(the plotting step)
creates a Dot Plot. It helps visualize which cell types are communicating and how "strong" or "significant" those messages are for a specific disease stage.
```
# ── 3. Dot plot: top LR pairs per stage ──────────────────────────────────────
def plot_lr_dotplot(liana_res: pd.DataFrame, stage: str, n_top: int = 15):
    sub = liana_res[liana_res["stage"] == stage].copy()
    if sub.empty:
        return
    sub = sub.sort_values("aggregate_rank").head(n_top)

    # creates a long string for the X-axis labels that tells you everything:
    # Who? (Source Cell → Target Cell)
    # How? (Ligand — Receptor)
    # Example: Ductal Cell → T-Cell \n CXCL12 — CXCR4
    sub["interaction"] = sub["source"] + " → " + sub["target"] + "\n" + \
                         sub["ligand_complex"] + " — " + sub["receptor_complex"]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        sub["interaction"],
        sub["cellphone_pvals"],
        c=-np.log10(sub["aggregate_rank"] + 1e-10),        # Consensus Score: Darker reds indicate a higher ranking(log-transform) across all LIANA methods.
        s=sub["lr_means"] * 30,                            # Interaction Magnitude: bigger dot means the ligand and receptor are expressed at higher levels.
        cmap="Reds", edgecolors="grey", linewidths=0.5,
    )

    #graph labeling and outputting 
    plt.colorbar(scatter, ax=ax, label="-log10(aggregate rank)")
    ax.set_xticklabels(sub["interaction"], rotation=90, fontsize=7)
    ax.set_ylabel("CellPhoneDB p-value")
    ax.set_title(f"Top LR interactions — {stage}")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/liana_dotplot_{stage}.png", dpi=150)
    plt.close()
```
## 3. Squidpy Spatial Co-occurrence
* **Purpose**: Validates if the cells "talking" via LIANA are actually standing near each other.
* **Mechanism**:
  *_sq.gr.spatial_neighbors_: Builds a physical graph of the tissue.
  * _sq.gr.co_occurrence_: Calculates the probability of finding Cell Type B at certain distances from Cell Type A.
* **Insight**: If LIANA says "Ductal cells talk to CAFs," but Co-occurrence shows they are never near each other, the interaction is likely false.
```
# ── 4. Squidpy spatial co-occurrence ─────────────────────────────────────────
def run_spatial_cooccurrence(adata: ad.AnnData, stage: str):
    """
    Compute spatial co-occurrence scores between cell types using Squidpy.
    Requires that cell-type annotations are in adata.obs[CELL_TYPE_COL].
    """
    sub = adata[adata.obs["disease_stage"] == stage].copy()
    if sub.n_obs < 50:
        return

    sq.gr.spatial_neighbors(sub, coord_type="visium")
    sq.gr.co_occurrence(sub, cluster_key=CELL_TYPE_COL)

    sq.pl.co_occurrence(
        sub, cluster_key=CELL_TYPE_COL,
        clusters=["Ductal (malignant)", "Stellate/CAF", "Macrophage"],
        figsize=(8, 4),
    )
    plt.savefig(f"{FIGURE_DIR}/cooccurrence_{stage}.png", dpi=150)
    plt.close()
```
##vVisualization & Comparison
* **Dot Plots** (_plot_lr_dotplot_): Summarizes the top interactions for a specific stage.
  * **Dot Size**: Mean expression level of the pair.
  * **Dot Color**: Significance ($-log_{10}$ of the rank).
* **Heatmap** _(plot_interaction_heatmap_): Tracks how specific "conversations" change across the Stage Order (e.g., from Normal $\rightarrow$ PanIN $\rightarrow$ Metastasis).
* **Logic**: This identifies "stage-specific" interactions—signals that only appear once the tumor becomes malignant.
```
# ── 5. Stage-transition interaction heatmap ───────────────────────────────────
def plot_interaction_heatmap(liana_res: pd.DataFrame):
    """Heatmap of aggregate rank scores for key LR pairs across stages."""
    top_pairs = get_top_interactions(liana_res, n_top=25)
    top_pairs["lr_pair"] = (top_pairs["source"] + " → " + top_pairs["target"] +
                            " | " + top_pairs["ligand_complex"] + "/" +
                            top_pairs["receptor_complex"])

    pivot = liana_res.merge(
        top_pairs[["source", "target", "ligand_complex", "receptor_complex"]],
        on=["source", "target", "ligand_complex", "receptor_complex"]
    )
    pivot["lr_pair"] = (pivot["source"] + " → " + pivot["target"] +
                        " | " + pivot["ligand_complex"] + "/" +
                        pivot["receptor_complex"])
    mat = pivot.pivot_table(
        index="lr_pair", columns="stage", values="aggregate_rank", aggfunc="min"
    ).reindex(columns=STAGE_ORDER)

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(-np.log10(mat + 1e-10), cmap="YlOrRd", ax=ax,
                linewidths=0.3, xticklabels=True, yticklabels=True)
    ax.set_title("LR interaction scores across disease stages\n(-log10 aggregate rank)")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/lr_stage_heatmap.png", dpi=150)
    plt.close()
```
## Main
run everything
```
# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading atlas with programs...")
    adata = sc.read_h5ad("data/integrated/pdac_atlas_with_programs.h5ad")

    print("Running LIANA across all stages...")
    liana_res = run_liana_all_stages(adata)

    print("Plotting per-stage LR dot plots...")
    for stage in STAGE_ORDER:
        plot_lr_dotplot(liana_res, stage)

    print("Running Squidpy spatial co-occurrence...")
    for stage in STAGE_ORDER:
        run_spatial_cooccurrence(adata, stage)

    print("Plotting cross-stage interaction heatmap...")
    plot_interaction_heatmap(liana_res)

    print("\nDone. Results in results/cci/ and figures/cci/")
````
