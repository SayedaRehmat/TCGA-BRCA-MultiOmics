"""
==============================================================================
STEP 1: DATA LOADING AND PREPROCESSING
Project: Multi-omics Prognostic Subtype Discovery in Breast Cancer (TCGA-BRCA)
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gzip
import pickle

# =============================================================================
# PROJECT PATHS
# =============================================================================
BASE_DIR = "/content/drive/MyDrive/TCGA_Project"

TCGA_EXPR_PATH = os.path.join(BASE_DIR, "gene_expression.tsv")
TCGA_CLIN_PATH = os.path.join(BASE_DIR, "clinical_survival.tsv")

OUTDIR = os.path.join(BASE_DIR, "results")
FIGDIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# ── 1. Load clinical data ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: DATA LOADING & PREPROCESSING")
print("=" * 60)

clin = pd.read_csv(TCGA_CLIN_PATH, sep="\t", index_col=0)

clin['sample_clean'] = clin['sample'].str.replace(
    r'-01$', '', regex=True
)

print(f"[Clinical] {clin.shape[0]} patients, {clin.shape[1]} columns")

# ── 2. Load gene expression matrix ─────────────────────────────────────────
print("Loading gene expression matrix (this takes ~30s)...")

expr = pd.read_csv(
    TCGA_EXPR_PATH,
    sep='\t',
    index_col=0
)

expr.columns = [c[:15] for c in expr.columns]

print(f"[Expression] {expr.shape[0]} genes × {expr.shape[1]} samples")

# ── 3. Match samples ───────────────────────────────────────────────────────
clin['barcode15'] = clin['sample'].str[:15]

common = sorted(
    set(clin['barcode15']) & set(expr.columns)
)

print(f"[Overlap] {len(common)} matched samples")

expr_matched = expr[common].T

clin_matched = (
    clin
    .set_index('barcode15')
    .loc[common]
    .copy()
)

clin_matched = clin_matched[
    clin_matched['OS.time'].notna()
]

expr_matched = expr_matched.loc[clin_matched.index]

print(f"[After OS filter] {clin_matched.shape[0]} usable samples")

# ── 4. Gene filtering: remove low-variance genes ──────────────────────────
gene_var = expr_matched.var()

threshold = gene_var.quantile(0.25)

expr_filt = expr_matched.loc[
    :,
    gene_var > threshold
]

print(
    f"[Gene filter] "
    f"{expr_filt.shape[1]} high-variance genes retained"
)

# ── 5. Summary statistics ─────────────────────────────────────────────────
summary = {
    "n_patients": int(clin_matched.shape[0]),
    "n_genes": int(expr_filt.shape[1]),
    "n_events_OS": int(clin_matched['OS'].sum()),
    "median_OS_days": float(clin_matched['OS.time'].median()),
    "pct_events": round(clin_matched['OS'].mean() * 100, 1),
}

print("\n[Summary]")

for k, v in summary.items():
    print(f"  {k:25s}: {v}")

# ── 6. Save processed objects ─────────────────────────────────────────────
EXPR_SAVE_PATH = os.path.join(OUTDIR, "expr_filtered.pkl")
CLIN_SAVE_PATH = os.path.join(OUTDIR, "clin_matched.pkl")
SUMMARY_SAVE_PATH = os.path.join(OUTDIR, "summary_stats.csv")

with open(EXPR_SAVE_PATH, 'wb') as f:
    pickle.dump(expr_filt, f)

with open(CLIN_SAVE_PATH, 'wb') as f:
    pickle.dump(clin_matched, f)

pd.DataFrame([summary]).to_csv(
    SUMMARY_SAVE_PATH,
    index=False
)

print(f"[Saved] {EXPR_SAVE_PATH}")
print(f"[Saved] {CLIN_SAVE_PATH}")
print(f"[Saved] {SUMMARY_SAVE_PATH}")

# ── 7. QC Figure: OS time distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

fig.suptitle(
    "TCGA-BRCA: Clinical Data QC",
    fontsize=14,
    fontweight='bold'
)

# OS distribution
axes[0].hist(
    clin_matched['OS.time'] / 365,
    bins=30,
    color='steelblue',
    edgecolor='white'
)

axes[0].set_xlabel(
    "Overall Survival (years)",
    fontsize=12
)

axes[0].set_ylabel(
    "Number of Patients",
    fontsize=12
)

axes[0].set_title(
    f"OS Distribution (n={clin_matched.shape[0]})"
)

axes[0].axvline(
    clin_matched['OS.time'].median() / 365,
    color='red',
    linestyle='--',
    label=(
        f"Median="
        f"{clin_matched['OS.time'].median()/365:.1f}y"
    )
)

axes[0].legend()

# Event distribution
event_counts = (
    clin_matched['OS']
    .value_counts()
    .sort_index()
)

axes[1].bar(
    ['Censored (0)', 'Death (1)'],
    event_counts.values,
    color=['#4CAF50', '#F44336'],
    edgecolor='white',
    width=0.5
)

for i, v in enumerate(event_counts.values):
    axes[1].text(
        i,
        v + 5,
        str(v),
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

axes[1].set_ylabel(
    "Count",
    fontsize=12
)

axes[1].set_title(
    "Event Distribution"
)

plt.tight_layout()

FIG_SAVE_PATH = os.path.join(
    FIGDIR,
    "01_clinical_QC.png"
)

plt.savefig(
    FIG_SAVE_PATH,
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print(f"\n[Saved] {FIG_SAVE_PATH}")

print("STEP 1 COMPLETE ✓")
