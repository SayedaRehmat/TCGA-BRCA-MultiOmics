"""
==============================================================================
STEP 4: DIFFERENTIAL EXPRESSION & GENE SIGNATURE DISCOVERY
  - One-vs-rest t-test per cluster (FDR correction)
  - Top DEGs per cluster
  - PAM50 gene overlap validation
  - Heatmap of top signature genes
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats

OUTDIR = "/home/claude/BRCA_project/results"
FIGDIR = "/home/claude/BRCA_project/figures"

print("=" * 60)
print("STEP 4: DIFFERENTIAL EXPRESSION & GENE SIGNATURES")
print("=" * 60)

with open(f"{OUTDIR}/expr_filtered.pkl", 'rb') as f:
    expr = pickle.load(f)
with open(f"{OUTDIR}/clin_clustered.pkl", 'rb') as f:
    clin = pickle.load(f)

clusters = sorted(clin['cluster_label'].unique())
n_cl = len(clusters)

# ── PAM50 Genes (Parker et al. 2009, validated set) ──────────────────────────
PAM50_GENES = [
    'ACTR3B','ANLN','BAG1','BCL2','BIRC5','BLVRA','CCNB1','CCNE1',
    'CDC20','CDC6','CDH3','CENPF','CEP55','CXXC5','EGFR','ERBB2',
    'ESR1','EXO1','FGFR4','FOXA1','FOXC1','GPR160','GRB7','KIF2C',
    'KRT14','KRT17','KRT5','MAPT','MDM2','MELK','MIA','MKI67','MLPH',
    'MMP11','MYBL2','MYC','NAT1','NDC80','NUF2','ORC6','PGR','PHGDH',
    'PTTG1','RRM2','SFRP1','SLC39A6','TMEM45B','TYMS','UBE2C','UBE2T'
]

# ── One-vs-rest differential expression ──────────────────────────────────────
print("\nRunning one-vs-rest differential expression tests...")

def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.arange(1, n+1)
    q = pvals[order] * n / ranked
    q = np.minimum.accumulate(q[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(q, 1.0)
    return adj

all_degs = {}
for cl in clusters:
    in_cl  = clin['cluster_label'] == cl
    group1 = expr.loc[in_cl]
    group2 = expr.loc[~in_cl]

    # t-test per gene
    t_stats, p_vals = stats.ttest_ind(group1.values, group2.values,
                                       axis=0, equal_var=False)
    fc = group1.mean(axis=0) - group2.mean(axis=0)   # log2 fold-change (already log2)

    result = pd.DataFrame({
        'gene'    : expr.columns,
        't_stat'  : t_stats,
        'p_value' : p_vals,
        'log2FC'  : fc,
    })
    result['p_adj'] = fdr_bh(np.abs(p_vals))
    result = result.sort_values('p_value')
    sig = result[(result['p_adj'] < 0.05) & (result['log2FC'].abs() > 0.5)]
    up  = sig[sig['log2FC'] > 0]
    dn  = sig[sig['log2FC'] < 0]
    print(f"  {cl}: {len(up)} up-regulated, {len(dn)} down-regulated (FDR<5%, |FC|>0.5)")
    all_degs[cl] = result
    result.to_csv(f"{OUTDIR}/DEG_{cl}.csv", index=False)

# ── PAM50 overlap ─────────────────────────────────────────────────────────────
print("\nPAM50 gene availability in expression matrix:")
pam50_avail = [g for g in PAM50_GENES if g in expr.columns]
print(f"  {len(pam50_avail)}/{len(PAM50_GENES)} PAM50 genes found")

print("\nPAM50 differential expression per cluster (top hits):")
pam50_summary = {}
for cl in clusters:
    deg = all_degs[cl].set_index('gene')
    pam50_deg = deg.loc[[g for g in pam50_avail if g in deg.index]]
    top10 = pam50_deg.nsmallest(10, 'p_value')[['log2FC', 'p_value', 'p_adj']]
    pam50_summary[cl] = top10
    print(f"\n  {cl} - Top PAM50 genes:")
    print(top10.round(4).to_string())

# ── Heatmap of top 50 DEGs ────────────────────────────────────────────────────
# Collect top 50 unique genes across all clusters
top_genes_per_cl = {}
for cl in clusters:
    deg = all_degs[cl]
    sig = deg[(deg['p_adj'] < 0.05) & (deg['log2FC'] > 0.5)]
    top_genes_per_cl[cl] = sig.nsmallest(50, 'p_value')['gene'].tolist()

all_top = []
for g_list in top_genes_per_cl.values():
    all_top.extend(g_list)
heatmap_genes = list(dict.fromkeys(all_top))[:100]  # deduplicate, keep order

# Subsample patients for heatmap (max 200 per cluster)
sub_idx = []
for cl in clusters:
    idx = clin[clin['cluster_label'] == cl].index.tolist()
    sub_idx.extend(idx[:min(200, len(idx))])

hmap_data = expr.loc[sub_idx, [g for g in heatmap_genes if g in expr.columns]]
hmap_data = hmap_data.T

# Z-score each gene
from scipy.stats import zscore
hmap_z = hmap_data.apply(zscore, axis=1).fillna(0)
# hmap_z: rows=genes, columns=patient barcodes

# Cluster annotation
cluster_col = clin.loc[sub_idx, 'cluster_label']
color_map = {'C1': '#1f77b4', 'C2': '#d62728', 'C3': '#2ca02c',
             'C4': '#9467bd', 'C5': '#8c564b'}

fig, ax = plt.subplots(figsize=(16, 10))
# Sort columns (patients) by cluster
sorted_cols = cluster_col.sort_values().index   # patient barcodes sorted by cluster
hmap_sorted = hmap_z[sorted_cols]               # reorder patient columns

im = ax.imshow(hmap_sorted.values, aspect='auto', cmap='RdBu_r',
               vmin=-2, vmax=2, interpolation='none')
plt.colorbar(im, ax=ax, label='Z-score expression', shrink=0.6)

# Mark cluster boundaries
boundary = 0
for cl in clusters:
    n = (cluster_col == cl).sum()
    ax.axvline(boundary + n - 0.5, color='black', linewidth=1.5)
    ax.text(boundary + n/2, -2, cl, ha='center', va='top', fontsize=10,
            fontweight='bold', color=color_map.get(cl, 'black'))
    boundary += n

ax.set_yticks(range(len(hmap_sorted.index)))
ax.set_yticklabels(hmap_sorted.index, fontsize=6)
ax.set_xticks([])
ax.set_title("Gene Expression Heatmap — Top DEGs per Cluster\n(Z-scored, sorted by cluster)",
             fontsize=13, fontweight='bold')
ax.set_ylabel("Genes", fontsize=11)
ax.set_xlabel("Patients (sorted by cluster)", fontsize=11)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/04_heatmap_DEGs.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] {FIGDIR}/04_heatmap_DEGs.png")

# ── Volcano plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, n_cl, figsize=(6*n_cl, 5))
if n_cl == 1: axes = [axes]
fig.suptitle("Volcano Plots — One-vs-Rest Differential Expression", fontsize=13, fontweight='bold')

for ax, cl in zip(axes, clusters):
    deg = all_degs[cl].copy()
    deg['-log10p'] = -np.log10(deg['p_value'].clip(1e-300))
    sig_mask = (deg['p_adj'] < 0.05) & (deg['log2FC'].abs() > 0.5)
    up_mask  = sig_mask & (deg['log2FC'] > 0)
    dn_mask  = sig_mask & (deg['log2FC'] < 0)

    ax.scatter(deg.loc[~sig_mask, 'log2FC'], deg.loc[~sig_mask, '-log10p'],
               s=2, alpha=0.3, color='gray', label='NS')
    ax.scatter(deg.loc[up_mask, 'log2FC'], deg.loc[up_mask, '-log10p'],
               s=6, alpha=0.6, color='#d62728', label=f'Up ({up_mask.sum()})')
    ax.scatter(deg.loc[dn_mask, 'log2FC'], deg.loc[dn_mask, '-log10p'],
               s=6, alpha=0.6, color='#1f77b4', label=f'Down ({dn_mask.sum()})')

    # Label top 10
    top10 = deg[sig_mask].nsmallest(10, 'p_value')
    for _, row in top10.iterrows():
        ax.text(row['log2FC'], row['-log10p'], row['gene'],
                fontsize=6, ha='center', va='bottom', alpha=0.8)

    ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0.5,  color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(-0.5, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("log2 Fold Change", fontsize=11)
    ax.set_ylabel("-log10(p-value)", fontsize=11)
    ax.set_title(f"Cluster {cl}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/04b_volcano.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved] {FIGDIR}/04b_volcano.png")
print("\nSTEP 4 COMPLETE ✓")
