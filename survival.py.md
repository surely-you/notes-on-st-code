# Annotations on script_survival.py
take "gene programs" (discovered via spatial transcriptomics) and prove they have clinical relevance by showing they can predict how long a patient will live using large-scale patient databases (TCGA)

## summary
* Input: A list of gene programs (sets of genes that tend to work together).
* Scoring: It calculates how "active" those programs are in hundreds of pancreatic cancer patients.
* Testing: It checks if patients with "High" program activity die sooner than those with "Low" activity.
* Verification: It repeats this across six different datasets (TCGA plus 5 GEO cohorts) to ensure the findings aren't a fluke.

## set up enviroment and stuff
new libraries/functions
- **zscore**
- **KaplanMeierFitter**
- **CoxPHFitter**
- **logrank_test**
- **StandardScaler**
```
"""
06_survival_validation.py
Validate spatially-derived gene signatures against TCGA-PAAD bulk RNA-seq
survival data, with replication in external cohorts.

Workflow:
  1. Score each TCGA sample for spatial gene programs (ssGSEA / mean z-score)
  2. Dichotomize patients by program score (median split or optimal cutpoint)
  3. Kaplan-Meier survival curves + log-rank test
  4. Multivariate Cox proportional hazards (age, stage, grade, program score)
  5. Validate in GSE39582, GSE17538, GSE29621, GSE72970, GSE161158

Dependencies: pandas, numpy, scipy, lifelines, sklearn, matplotlib, pydeseq2 (optional)
Note: TCGA data downloaded via GDCquery (R/TCGAbiolinks) or cBioPortal bulk export.
      External GEO cohorts: download normalized expression + clinical metadata.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "results/survival"
FIGURE_DIR = "figures/survival"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# External validation cohorts (GEO accessions)
VALIDATION_COHORTS = ["GSE39582", "GSE17538", "GSE29621", "GSE72970", "GSE161158"]

# Columns expected in clinical metadata files
SURV_TIME_COL   = "OS.time"   # overall survival time (days)
SURV_EVENT_COL  = "OS"        # event: 1=death, 0=censored


# ── 1. Load NMF program gene sets ────────────────────────────────────────────
def load_programs(programs_csv: str) -> dict[str, list[str]]:
    df = pd.read_csv(programs_csv)
    programs = {}
    for _, row in df.iterrows():
        prog   = row["program"]
        genes  = [g.strip() for g in row["top_genes"].split(",")]
        programs[prog] = genes
    return programs
```
## Converts a list of genes into a single "activity score" for every patient.
Genes have different natural scales. Z-scoring ensures a highly expressed gene doesn't "drown out" a lowly expressed gene in the same program. The final score is the mean Z-score of all genes in that program.
```
# ── 2. Score bulk RNA-seq samples with gene programs (mean z-score) ───────────
def score_programs(expr: pd.DataFrame, programs: dict[str, list[str]]) -> pd.DataFrame:
    """
    expr: genes × samples (log-normalized TPM or FPKM)
    Returns: samples × program_scores
    """
    expr_z = expr.apply(zscore, axis=1)   # z-score each gene across samples
    scores = {}
    for prog, genes in programs.items():
        avail = [g for g in genes if g in expr_z.index]
        if len(avail) < 5:
            print(f"  Warning: {prog} has only {len(avail)} genes in bulk data")
        scores[prog] = expr_z.loc[avail].mean(axis=0)
    return pd.DataFrame(scores)
```
## Probability: Visualizes survival over time.
splits patients into two groups (High vs. Low) based on the median program score.
* The Math: It uses the Kaplan-Meier Estimator to calculate the probability of surviving at time $t$:
* $$S(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$$
  * $d_i$: Number of deaths at time $t$.
  * $n_i$: Number of people still alive (at risk) just before time $t$.
* Log-rank Test: It produces a $p$-value to determine if the difference between the red (High) and blue (Low) curves is statistically significant.
```
# ── 3. Kaplan-Meier analysis ──────────────────────────────────────────────────
def kaplan_meier(clin: pd.DataFrame, prog_scores: pd.DataFrame,
                 program: str, cohort_name: str):
    """Dichotomize by median, plot KM curves, report log-rank p-value."""
    df = clin[[SURV_TIME_COL, SURV_EVENT_COL]].join(prog_scores[[program]]).dropna()
    median_score = df[program].median()
    df["group"]  = np.where(df[program] >= median_score, "High", "Low")

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"High": "#e74c3c", "Low": "#3498db"}
    for grp, color in colors.items():
        mask = df["group"] == grp
        kmf.fit(df.loc[mask, SURV_TIME_COL] / 365,
                df.loc[mask, SURV_EVENT_COL], label=grp)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    # Log-rank test
    res = logrank_test(
        df.loc[df["group"] == "High", SURV_TIME_COL],
        df.loc[df["group"] == "Low",  SURV_TIME_COL],
        df.loc[df["group"] == "High", SURV_EVENT_COL],
        df.loc[df["group"] == "Low",  SURV_EVENT_COL],
    )
    ax.set_title(f"{cohort_name} | {program}\nLog-rank p = {res.p_value:.4f}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/KM_{cohort_name}_{program}.png", dpi=150)
    plt.close()
    return {"cohort": cohort_name, "program": program,
            "logrank_p": res.p_value, "median_high": df.loc[df["group"]=="High", SURV_TIME_COL].median(),
            "median_low": df.loc[df["group"]=="Low", SURV_TIME_COL].median()}
```
Cox Proportional Hazards Model:
* $$h(t|X) = h_0(t) \exp(\sum_{i=1}^{n} \beta_i X_i)$$
  * $h(t|X)$ is the risk (hazard).
  * $\beta_i$ is the Coefficient.
  * If $\beta$ is positive, higher expression = higher risk of death.
* uses **StandardScaler** to ensure coefficients are comparable, allowing you to say "a 1-standard-deviation increase in program score increases risk by $X\%$."
```
# ── 4. Multivariate Cox PH ────────────────────────────────────────────────────
def cox_multivariate(clin: pd.DataFrame, prog_scores: pd.DataFrame,
                     program: str, covariates: list[str], cohort_name: str):
    """
    Fit Cox PH with program score + clinical covariates.
    covariates: list of column names in clin (e.g. ['age', 'stage_num', 'grade'])
    """
    df = clin[[SURV_TIME_COL, SURV_EVENT_COL] + covariates] \
             .join(prog_scores[[program]]).dropna()

    # Standardize continuous variables
    for col in [program] + covariates:
        df[col] = StandardScaler().fit_transform(df[[col]])

    cph = CoxPHFitter()
    cph.fit(df, duration_col=SURV_TIME_COL, event_col=SURV_EVENT_COL)
    cph.print_summary()
    cph.plot()
    plt.title(f"Cox PH — {cohort_name} | {program}")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/Cox_{cohort_name}_{program}.png", dpi=150)
    plt.close()
    return cph.summary
```
```
# ── 5. Forest plot across validation cohorts ──────────────────────────────────
def forest_plot(cox_summaries: dict[str, pd.DataFrame], program: str):
    """
    cox_summaries: {cohort_name: cph.summary DataFrame}
    Plots HR ± 95% CI for `program` across cohorts.
    """
    records = []
    for cohort, summ in cox_summaries.items():
        if program not in summ.index:
            continue
        row = summ.loc[program]
        records.append({
            "cohort": cohort,
            "HR"    : np.exp(row["coef"]),
            "HR_lo" : np.exp(row["coef lower 95%"]),
            "HR_hi" : np.exp(row["coef upper 95%"]),
            "p"     : row["p"],
        })
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(6, len(df) * 0.6 + 1))
    for i, row in df.iterrows():
        ax.plot([row["HR_lo"], row["HR_hi"]], [i, i], color="steelblue", lw=2)
        ax.scatter(row["HR"], i, color="steelblue", zorder=5, s=60)
        ax.text(row["HR_hi"] + 0.05, i, f"p={row['p']:.3f}", va="center", fontsize=8)
    ax.axvline(1, color="grey", linestyle="--", lw=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["cohort"])
    ax.set_xlabel("Hazard Ratio (HR)")
    ax.set_title(f"Forest plot — {program}")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/forest_{program}.png", dpi=150)
    plt.close()
    df.to_csv(f"{OUTPUT_DIR}/forest_{program}.csv", index=False)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    programs = load_programs("results/disease_stage/nmf_programs.csv")

    # ── TCGA-PAAD ────────────────────────────────────────────────────────────
    print("Loading TCGA-PAAD data...")
    tcga_expr  = pd.read_csv("data/survival/TCGA_PAAD_expr.csv", index_col=0)
    tcga_clin  = pd.read_csv("data/survival/TCGA_PAAD_clinical.csv", index_col=0)
    tcga_scores = score_programs(tcga_expr, programs)
    tcga_scores.index = tcga_clin.index

    km_results    = []
    cox_summaries = {}

    for prog in programs:
        print(f"  KM + Cox for {prog} in TCGA-PAAD...")
        km_results.append(
            kaplan_meier(tcga_clin, tcga_scores, prog, "TCGA-PAAD")
        )
        summ = cox_multivariate(
            tcga_clin, tcga_scores, prog,
            covariates=["age_at_index", "ajcc_pathologic_stage_num"],
            cohort_name="TCGA-PAAD",
        )
        cox_summaries["TCGA-PAAD"] = summ

    # ── External validation cohorts ───────────────────────────────────────────
    for cohort in VALIDATION_COHORTS:
        expr_path = f"data/survival/{cohort}_expr.csv"
        clin_path = f"data/survival/{cohort}_clinical.csv"
        if not (os.path.exists(expr_path) and os.path.exists(clin_path)):
            print(f"  Skipping {cohort} — files not found")
            continue
        print(f"  Validating in {cohort}...")
        expr   = pd.read_csv(expr_path, index_col=0)
        clin   = pd.read_csv(clin_path, index_col=0)
        scores = score_programs(expr, programs)
        scores.index = clin.index
        for prog in programs:
            km_results.append(kaplan_meier(clin, scores, prog, cohort))
            summ = cox_multivariate(clin, scores, prog,
                                    covariates=["age"], cohort_name=cohort)
            cox_summaries[cohort] = summ

    pd.DataFrame(km_results).to_csv(f"{OUTPUT_DIR}/km_summary.csv", index=False)

    # ── Forest plots ──────────────────────────────────────────────────────────
    for prog in programs:
        forest_plot(cox_summaries, prog)

    print("\nDone. Results in results/survival/ and figures/survival/")
```
