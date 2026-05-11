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


# ── 1. LIANA ligand-receptor analysis ────────────────────────────────────────
def run_liana(adata: ad.AnnData, stage: str) -> pd.DataFrame:
    """Run LIANA consensus LR scoring for one disease stage subset."""
    sub = adata[adata.obs["disease_stage"] == stage].copy()
    if sub.n_obs < 50:
        print(f"  {stage}: too few spots, skipping")
        return pd.DataFrame()

    liana.mt.rank_aggregate(
        sub,
        groupby=CELL_TYPE_COL,
        use_raw=True,
        verbose=False,
    )
    res = sub.uns["liana_res"].copy()
    res["stage"] = stage
    return res


def run_liana_all_stages(adata: ad.AnnData) -> pd.DataFrame:
    all_res = []
    for stage in STAGE_ORDER:
        print(f"  Running LIANA for {stage}...")
        df = run_liana(adata, stage)
        all_res.append(df)
    combined = pd.concat(all_res, ignore_index=True)
    combined.to_csv(f"{OUTPUT_DIR}/liana_all_stages.csv", index=False)
    return combined


# ── 2. Filter & rank interactions ─────────────────────────────────────────────
def get_top_interactions(liana_res: pd.DataFrame, n_top: int = 20) -> pd.DataFrame:
    """Filter to significant interactions and rank by aggregate rank score."""
    sig = liana_res[liana_res["aggregate_rank"] < 0.05].copy()
    ranked = (
        sig.groupby(["source", "target", "ligand_complex", "receptor_complex"])
        .agg(mean_rank=("aggregate_rank", "mean"), n_stages=("stage", "nunique"))
        .reset_index()
        .sort_values("mean_rank")
    )
    ranked.to_csv(f"{OUTPUT_DIR}/top_interactions_consensus.csv", index=False)
    return ranked.head(n_top)


# ── 3. Dot plot: top LR pairs per stage ──────────────────────────────────────
def plot_lr_dotplot(liana_res: pd.DataFrame, stage: str, n_top: int = 15):
    sub = liana_res[liana_res["stage"] == stage].copy()
    if sub.empty:
        return
    sub = sub.sort_values("aggregate_rank").head(n_top)
    sub["interaction"] = sub["source"] + " → " + sub["target"] + "\n" + \
                         sub["ligand_complex"] + " — " + sub["receptor_complex"]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        sub["interaction"],
        sub["cellphone_pvals"],
        c=-np.log10(sub["aggregate_rank"] + 1e-10),
        s=sub["lr_means"] * 30,
        cmap="Reds", edgecolors="grey", linewidths=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="-log10(aggregate rank)")
    ax.set_xticklabels(sub["interaction"], rotation=90, fontsize=7)
    ax.set_ylabel("CellPhoneDB p-value")
    ax.set_title(f"Top LR interactions — {stage}")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/liana_dotplot_{stage}.png", dpi=150)
    plt.close()


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
