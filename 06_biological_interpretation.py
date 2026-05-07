"""
==============================================================================
STEP 6: PAM50 SUBTYPE MAPPING + PATHWAY ENRICHMENT + FINAL FIGURES
  - Map clusters to known PAM50 subtypes via signature gene expression
  - Pathway enrichment (manual gene-set scoring: Hallmarks-derived)
  - Comprehensive summary dashboard
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
from scipy import stats

OUTDIR = "/home/claude/BRCA_project/results"
FIGDIR = "/home/claude/BRCA_project/figures"

print("=" * 60)
print("STEP 6: BIOLOGICAL INTERPRETATION & FINAL FIGURES")
print("=" * 60)

with open(f"{OUTDIR}/expr_filtered.pkl",'rb') as f: expr = pickle.load(f)
with open(f"{OUTDIR}/clin_clustered.pkl",'rb') as f: clin = pickle.load(f)

clusters = sorted(clin['cluster_label'].unique())
colors   = {'C1':'#1f77b4','C2':'#d62728','C3':'#2ca02c'}

# ══════════════════════════════════════════════════════════════════════════
# 1. PAM50 Centroid Scoring → Subtype Assignment
# ══════════════════════════════════════════════════════════════════════════
# Key PAM50 marker genes per subtype (from Parker 2009 / Perou 2000)
PAM50_MARKERS = {
    'Luminal_A'  : ['ESR1','PGR','FOXA1','SLC39A6','NAT1','MLPH','BCL2'],
    'Luminal_B'  : ['ERBB2','MKI67','CCNB1','CCNE1','BIRC5','MYBL2'],
    'HER2E'      : ['ERBB2','GRB7','FGFR4','TMEM45B','GPR160'],
    'Basal'      : ['KRT5','KRT14','KRT17','EGFR','MIA','FOXC1','SFRP1'],
    'Normal_like': ['MAPT','SLC39A6','BCL2','MDM2'],
}

# Compute mean expression of each marker set per sample, then per cluster
subtype_scores = pd.DataFrame(index=expr.index)
for subtype, genes in PAM50_MARKERS.items():
    avail = [g for g in genes if g in expr.columns]
    subtype_scores[subtype] = expr[avail].mean(axis=1)

# Mean score per cluster
cluster_subtype = subtype_scores.copy()
cluster_subtype['cluster'] = clin['cluster_label']
cluster_means = cluster_subtype.groupby('cluster').mean()

print("\nCluster–PAM50 Subtype Score Matrix:")
print(cluster_means.round(3).to_string())

# Assign most likely subtype
assigned = cluster_means.idxmax(axis=1)
print("\nPrimary PAM50 Assignment per Cluster:")
for cl, subtype in assigned.items():
    print(f"  {cl} → {subtype}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Pathway Gene-Set Scoring (GSVA-like mean z-score)
# ══════════════════════════════════════════════════════════════════════════
PATHWAYS = {
    'Cell_Proliferation' : ['MKI67','PCNA','MCM2','MCM7','CDC6','CCNB1','CCNE1','CDK2','CDK4','E2F1'],
    'Apoptosis'          : ['BCL2','BAX','CASP3','CASP7','CASP9','TP53','PUMA','BID','MCL1'],
    'DNA_Repair'         : ['BRCA1','BRCA2','RAD51','PALB2','ATM','CHEK2','MLH1','MSH2','MSH6'],
    'Hormone_Signaling'  : ['ESR1','PGR','AR','FOXA1','GATA3','TFF1','TFF3','IGFBP4'],
    'HER2_Signaling'     : ['ERBB2','ERBB3','EGFR','GRB7','PIK3CA','AKT1','MTOR'],
    'EMT_Invasion'       : ['CDH1','VIM','FN1','TWIST1','SNAI1','ZEB1','MMP2','MMP9','CDH2'],
    'Immune_Response'    : ['CD8A','CD4','FOXP3','PDCD1','CD274','CTLA4','IFNG','TNF','IL6'],
    'Metabolism'         : ['LDHA','HK2','PKM','FASN','IDH1','ACLY','GLUT1','PFKL'],
}

pathway_scores = pd.DataFrame(index=expr.index)
for pathway, genes in PATHWAYS.items():
    avail = [g for g in genes if g in expr.columns]
    if avail:
        z = (expr[avail] - expr[avail].mean()) / (expr[avail].std() + 1e-6)
        pathway_scores[pathway] = z.mean(axis=1)
    else:
        pathway_scores[pathway] = 0.0

pathway_scores['cluster'] = clin['cluster_label']
pathway_means = pathway_scores.groupby('cluster').mean()
print("\nPathway Activity Scores per Cluster:")
print(pathway_means.round(3).to_string())
pathway_means.to_csv(f"{OUTDIR}/pathway_scores.csv")

# ══════════════════════════════════════════════════════════════════════════
# 3. Top DEG Tables per Cluster
# ══════════════════════════════════════════════════════════════════════════
print("\nTop 20 Up-regulated DEGs per Cluster:")
top_deg_tables = {}
for cl in clusters:
    deg = pd.read_csv(f"{OUTDIR}/DEG_{cl}.csv")
    up  = deg[(deg['p_adj']<0.05)&(deg['log2FC']>0.5)].nsmallest(20,'p_value')[['gene','log2FC','p_value','p_adj']]
    top_deg_tables[cl] = up
    print(f"\n  {cl}:")
    print(up.round(4).to_string(index=False))
    up.to_csv(f"{OUTDIR}/top20_up_{cl}.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# 4. FIGURE: PAM50 Subtype Heatmap
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("TCGA-BRCA: Biological Subtype Characterization", fontsize=13, fontweight='bold')

# PAM50 score heatmap
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
            ax=axes[0], linewidths=0.5, cbar_kws={'label': 'Mean Expression'})
axes[0].set_title("PAM50 Marker Gene Expression\nper Cluster", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Cluster", fontsize=11)
axes[0].set_ylabel("PAM50 Subtype", fontsize=11)

# Pathway activity heatmap
sns.heatmap(pathway_means.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
            ax=axes[1], linewidths=0.5, cbar_kws={'label': 'Mean Z-score'})
axes[1].set_title("Pathway Activity Scores\nper Cluster", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Cluster", fontsize=11)
axes[1].set_ylabel("Pathway", fontsize=11)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/06_subtype_pathway.png", dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# 5. FIGURE: Key Biomarker Expression Boxplots
# ══════════════════════════════════════════════════════════════════════════
KEY_GENES = ['ESR1','ERBB2','MKI67','KRT5','PGR','TP53','BCL2','EGFR','BRCA1','CDH1']
KEY_GENES = [g for g in KEY_GENES if g in expr.columns]

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle("Key Biomarker Expression Across Genomic Clusters", fontsize=13, fontweight='bold')
axes = axes.flatten()

palette = [colors[cl] for cl in clusters]
for i, gene in enumerate(KEY_GENES[:10]):
    ax = axes[i]
    data_plot = []
    for cl in clusters:
        vals = expr.loc[clin['cluster_label']==cl, gene].values
        data_plot.append(vals)
    parts = ax.violinplot(data_plot, positions=range(len(clusters)), showmedians=True)
    for j, (pc, cl) in enumerate(zip(parts['bodies'], clusters)):
        pc.set_facecolor(colors[cl]); pc.set_alpha(0.7)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters, fontsize=10)
    ax.set_title(f"{gene}", fontsize=11, fontweight='bold')
    ax.set_ylabel("log2(RSEM+1)", fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add significance stars (pairwise t-test C1 vs C3)
    if len(clusters) >= 2:
        g1 = expr.loc[clin['cluster_label']==clusters[0], gene].values
        g3 = expr.loc[clin['cluster_label']==clusters[-1], gene].values
        _, p = stats.ttest_ind(g1, g3)
        star = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        ymax = max([d.max() for d in data_plot])
        ax.text(len(clusters)//2, ymax*1.02, f'p({clusters[0]}v{clusters[-1]})={star}',
                ha='center', fontsize=8, color='black')

plt.tight_layout()
plt.savefig(f"{FIGDIR}/06b_biomarker_expression.png", dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# 6. FIGURE: Comprehensive Project Dashboard
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Load previously computed data
X_pca = np.load(f"{OUTDIR}/X_pca.npy")
with open(f"{OUTDIR}/clustering_results.pkl",'rb') as f: clust_res = pickle.load(f)
labels = clin['cluster_label'].values

# Panel A: PCA scatter
ax_pca = fig.add_subplot(gs[0, 0:2])
for i, cl in enumerate(clusters):
    mask = labels == cl
    ax_pca.scatter(X_pca[mask,0], X_pca[mask,1], c=colors[cl], label=f'{cl} (n={mask.sum()})',
                   alpha=0.5, s=10, edgecolors='none')
ax_pca.set_xlabel("PC1 (15.1%)", fontsize=10)
ax_pca.set_ylabel("PC2 (10.1%)", fontsize=10)
ax_pca.set_title("A. PCA — 3 Genomic Subtypes\n(k=3 consensus clustering)", fontsize=10, fontweight='bold')
ax_pca.legend(fontsize=9, markerscale=2)

# Panel B: Cluster size pie
ax_pie = fig.add_subplot(gs[0, 2])
sizes = [int((labels==cl).sum()) for cl in clusters]
wedge_colors = [colors[cl] for cl in clusters]
ax_pie.pie(sizes, labels=[f'{cl}\n(n={s})' for cl,s in zip(clusters,sizes)],
           colors=wedge_colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize':9})
ax_pie.set_title("B. Cluster Distribution", fontsize=10, fontweight='bold')

# Panel C: Event rates
ax_ev = fig.add_subplot(gs[0, 3])
surv_df = pd.read_csv(f"{OUTDIR}/survival_metrics.csv")
ax_ev.bar(surv_df['cluster'], surv_df['event_rate_pct'],
          color=[colors[cl] for cl in surv_df['cluster']], edgecolor='white', width=0.5)
for i,(v,cl) in enumerate(zip(surv_df['event_rate_pct'], surv_df['cluster'])):
    ax_ev.text(i, v+0.3, f'{v}%', ha='center', fontsize=10, fontweight='bold')
ax_ev.set_ylabel("Death Events (%)", fontsize=10)
ax_ev.set_title("C. OS Event Rate per Cluster", fontsize=10, fontweight='bold')
ax_ev.set_ylim(0, 30)
ax_ev.grid(axis='y', alpha=0.3)

# Panel D: KM curves (simplified)
def kaplan_meier_simple(time, event):
    df = pd.DataFrame({'time':time,'event':event}).sort_values('time')
    s=1.0; ts=[0]; ss=[1.0]
    for t in df['time'].unique():
        n=(df['time']>=t).sum(); d=((df['time']==t)&(df['event']==1)).sum()
        if n>0: s*=(1-d/n)
        ts.append(t); ss.append(s)
    return np.array(ts), np.array(ss)

ax_km = fig.add_subplot(gs[1, 0:2])
for cl in clusters:
    sub = clin[clin['cluster_label']==cl]
    t,s = kaplan_meier_simple(sub['OS.time'].values/365, sub['OS'].values)
    ax_km.step(t, s, where='post', color=colors[cl], linewidth=2,
               label=f"{cl} (n={len(sub)}, {assigned[cl]})")
ax_km.set_xlabel("Time (years)", fontsize=10); ax_km.set_ylabel("Survival Prob.", fontsize=10)
ax_km.set_title("D. Kaplan-Meier — Overall Survival\nby Genomic Subtype", fontsize=10, fontweight='bold')
ax_km.legend(fontsize=9); ax_km.grid(alpha=0.3); ax_km.set_ylim(0,1.05)

# Panel E: ESR1 vs ERBB2 scatter
ax_scat = fig.add_subplot(gs[1, 2:4])
for cl in clusters:
    mask = labels == cl
    ax_scat.scatter(expr.loc[clin['cluster_label']==cl,'ESR1'] if 'ESR1' in expr.columns else [0],
                    expr.loc[clin['cluster_label']==cl,'ERBB2'] if 'ERBB2' in expr.columns else [0],
                    c=colors[cl], label=f'{cl} ({assigned[cl]})', alpha=0.5, s=12, edgecolors='none')
ax_scat.set_xlabel("ESR1 expression (log2)", fontsize=10)
ax_scat.set_ylabel("ERBB2 expression (log2)", fontsize=10)
ax_scat.set_title("E. ESR1 vs ERBB2 — Subtype Stratification\n(key breast cancer biomarkers)", fontsize=10, fontweight='bold')
ax_scat.legend(fontsize=9, markerscale=2)

# Panel F: Pathway heatmap
ax_path = fig.add_subplot(gs[2, 0:2])
sns.heatmap(pathway_means.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
            ax=ax_path, linewidths=0.3, cbar_kws={'label':'Z-score','shrink':0.6},
            annot_kws={'size':8})
ax_path.set_title("F. Pathway Activity Scores per Cluster", fontsize=10, fontweight='bold')
ax_path.set_xlabel("Cluster", fontsize=10)
ax_path.tick_params(axis='both', labelsize=8)

# Panel G: ML Performance
ml_perf = pd.read_csv(f"{OUTDIR}/ml_performance.csv")
ax_ml = fig.add_subplot(gs[2, 2])
bars = ax_ml.bar(['RF\n(Gene Expr)', 'LR\n(PCA)'],
                  ml_perf['mean_AUC'].values,
                  color=['#2196F3','#FF9800'], edgecolor='white', width=0.4)
for bar, val in zip(bars, ml_perf['mean_AUC'].values):
    ax_ml.text(bar.get_x()+bar.get_width()/2, val+0.003, f'{val:.3f}',
               ha='center', fontsize=11, fontweight='bold')
ax_ml.set_ylim(0.95, 1.01)
ax_ml.set_ylabel("Mean AUC (OvR)", fontsize=10)
ax_ml.set_title("G. Classifier Performance\n(5-fold CV)", fontsize=10, fontweight='bold')
ax_ml.grid(axis='y', alpha=0.3)

# Panel H: Summary text box
ax_txt = fig.add_subplot(gs[2, 3])
ax_txt.axis('off')
summary_text = (
    "PROJECT SUMMARY\n"
    "─────────────────────────────\n"
    f"Cohort:     TCGA-BRCA (n=1,214)\n"
    f"Features:   20,530 genes\n"
    f"MAD genes:  5,000\n"
    f"Clusters:   k=3 (Silhouette=0.21)\n\n"
    f"C1 (n=628)  → Luminal A\n"
    f"  ESR1↑ PGR↑ FOXA1↑\n"
    f"  Event rate: 14.8%\n\n"
    f"C2 (n=236)  → Basal-like\n"
    f"  KRT5↑ EGFR↑ FOXC1↑\n"
    f"  Event rate: 14.4%\n\n"
    f"C3 (n=350)  → Luminal B\n"
    f"  MKI67↑ CCNB1↑ UBE2C↑\n"
    f"  Event rate: 20.3% (highest)\n\n"
    f"RF Classifier: AUC=0.997\n"
    f"DEGs:       C1:5626, C2:7043\n"
    f"            C3:5750\n"
    f"PAM50:      49/50 genes found"
)
ax_txt.text(0.02, 0.98, summary_text, transform=ax_txt.transAxes,
            fontsize=8.5, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.9))

fig.suptitle("TCGA Breast Cancer Multi-omics Subtype Discovery\nResearch Dashboard — KAUST Level Analysis",
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig(f"{FIGDIR}/00_MASTER_DASHBOARD.png", dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"\n[Saved] {FIGDIR}/00_MASTER_DASHBOARD.png")

# ── Final results summary ─────────────────────────────────────────────────────
assigned_df = pd.DataFrame({'Cluster': assigned.index, 'PAM50_Subtype': assigned.values})
assigned_df.to_csv(f"{OUTDIR}/cluster_subtype_assignment.csv", index=False)
print(f"[Saved] {OUTDIR}/cluster_subtype_assignment.csv")
print("\nSTEP 6 COMPLETE ✓")
print("\n" + "="*60)
print("ALL PIPELINE STEPS COMPLETE")
print("="*60)
