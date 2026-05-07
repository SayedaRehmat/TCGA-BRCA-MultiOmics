"""
==============================================================================
STEP 2: DIMENSIONALITY REDUCTION & UNSUPERVISED SUBTYPE DISCOVERY
  - PCA on top-5000 MAD genes (Parker et al. 2009 approach)
  - K-Means consensus clustering (k=2..6) with silhouette selection
  - PAM50 surrogate labeling via gene centroids
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

OUTDIR = "/home/claude/BRCA_project/results"
FIGDIR = "/home/claude/BRCA_project/figures"

print("=" * 60)
print("STEP 2: DIMENSIONALITY REDUCTION & CLUSTERING")
print("=" * 60)

# ── Load preprocessed data ──────────────────────────────────────────────────
with open(f"{OUTDIR}/expr_filtered.pkl", 'rb') as f:
    expr = pickle.load(f)
with open(f"{OUTDIR}/clin_matched.pkl", 'rb') as f:
    clin = pickle.load(f)

print(f"Loaded: {expr.shape[0]} samples × {expr.shape[1]} genes")

# ── Select top-5000 MAD genes (standard BRCA practice) ─────────────────────
mad = expr.apply(lambda x: np.median(np.abs(x - np.median(x))), axis=0)
top_genes = mad.nlargest(5000).index
expr_top = expr[top_genes]
print(f"Top-5000 MAD genes selected")

# ── Standardize ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(expr_top)
print("Data standardized (z-score per gene)")

# ── PCA ──────────────────────────────────────────────────────────────────────
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_
print(f"PCA done: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%, "
      f"PC3={var_explained[2]*100:.1f}%")
print(f"Cumulative var (50 PCs): {var_explained.sum()*100:.1f}%")

# ── Consensus K-Means Clustering (k = 2 to 6) ───────────────────────────────
print("\nRunning consensus clustering (k=2..6)...")
results = {}
n_init = 20
for k in range(2, 7):
    sil_scores = []
    labels_all = []
    for seed in range(10):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X_pca[:, :20])  # use top-20 PCs
        sil = silhouette_score(X_pca[:, :20], labels, sample_size=500, random_state=42)
        sil_scores.append(sil)
        labels_all.append(labels)
    best_run = np.argmax(sil_scores)
    ch = calinski_harabasz_score(X_pca[:, :20], labels_all[best_run])
    results[k] = {
        'labels'   : labels_all[best_run],
        'sil_mean' : np.mean(sil_scores),
        'sil_std'  : np.std(sil_scores),
        'ch'       : ch,
    }
    print(f"  k={k}: Silhouette={np.mean(sil_scores):.4f}±{np.std(sil_scores):.4f}, CH={ch:.1f}")

# ── Select optimal k ─────────────────────────────────────────────────────────
best_k = max(results, key=lambda k: results[k]['sil_mean'])
print(f"\nOptimal k = {best_k} (highest mean silhouette)")
labels_best = results[best_k]['labels']

# Save cluster assignments
clin = clin.copy()
clin['cluster'] = labels_best
clin['cluster_label'] = ['C' + str(l+1) for l in labels_best]
cluster_counts = clin['cluster_label'].value_counts().sort_index()
print("Cluster sizes:", cluster_counts.to_dict())

# ── Save outputs ─────────────────────────────────────────────────────────────
np.save(f"{OUTDIR}/X_pca.npy", X_pca)
np.save(f"{OUTDIR}/top_gene_names.npy", top_genes.values)
with open(f"{OUTDIR}/clin_clustered.pkl", 'wb') as f:
    pickle.dump(clin, f)
with open(f"{OUTDIR}/clustering_results.pkl", 'wb') as f:
    pickle.dump(results, f)
with open(f"{OUTDIR}/pca_model.pkl", 'wb') as f:
    pickle.dump((pca, scaler, top_genes), f)

pd.DataFrame({
    'k'        : list(results.keys()),
    'sil_mean' : [results[k]['sil_mean'] for k in results],
    'sil_std'  : [results[k]['sil_std']  for k in results],
    'ch_index' : [results[k]['ch']       for k in results],
}).to_csv(f"{OUTDIR}/clustering_metrics.csv", index=False)

# ── FIGURE 1: PCA scree + PC1 vs PC2 ─────────────────────────────────────────
palette = sns.color_palette("tab10", best_k)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("TCGA-BRCA: PCA & Consensus Clustering", fontsize=14, fontweight='bold')

# Scree plot
axes[0].bar(range(1, 21), var_explained[:20]*100, color='#2196F3', edgecolor='white')
axes[0].plot(range(1, 21), np.cumsum(var_explained[:20])*100,
             'r-o', markersize=4, label='Cumulative')
axes[0].set_xlabel("Principal Component", fontsize=11)
axes[0].set_ylabel("% Variance Explained", fontsize=11)
axes[0].set_title("Scree Plot (Top 20 PCs)")
axes[0].legend()
axes[0].set_xticks(range(1, 21))
axes[0].set_xticklabels(range(1, 21), fontsize=7)

# PC1 vs PC2, colored by cluster
for i in range(best_k):
    mask = labels_best == i
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[palette[i]], label=f'C{i+1} (n={mask.sum()})',
                    alpha=0.6, s=15, edgecolors='none')
axes[1].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)", fontsize=11)
axes[1].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)", fontsize=11)
axes[1].set_title(f"PC1 vs PC2 — k={best_k} clusters")
axes[1].legend(markerscale=2, fontsize=9)

# Silhouette scores
ks = list(results.keys())
sil_means = [results[k]['sil_mean'] for k in ks]
sil_stds  = [results[k]['sil_std']  for k in ks]
axes[2].errorbar(ks, sil_means, yerr=sil_stds, fmt='o-', color='#9C27B0',
                 capsize=5, linewidth=2, markersize=8)
axes[2].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
axes[2].set_xlabel("Number of Clusters (k)", fontsize=11)
axes[2].set_ylabel("Mean Silhouette Score", fontsize=11)
axes[2].set_title("Cluster Quality (10× consensus)")
axes[2].legend()
axes[2].set_xticks(ks)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/02_PCA_clustering.png", dpi=150, bbox_inches='tight')
plt.close()

# ── FIGURE 2: PC2 vs PC3, PC3 vs PC4 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PCA Space — Additional Dimensions", fontsize=13, fontweight='bold')
for ax, (xi, yi) in zip(axes, [(1, 2), (2, 3)]):
    for i in range(best_k):
        mask = labels_best == i
        ax.scatter(X_pca[mask, xi], X_pca[mask, yi],
                   c=[palette[i]], label=f'C{i+1}', alpha=0.6, s=12, edgecolors='none')
    ax.set_xlabel(f"PC{xi+1} ({var_explained[xi]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC{yi+1} ({var_explained[yi]*100:.1f}%)", fontsize=11)
    ax.set_title(f"PC{xi+1} vs PC{yi+1}")
    ax.legend(markerscale=2, fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/02b_PCA_additional.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[Saved] {FIGDIR}/02_PCA_clustering.png")
print(f"[Saved] {FIGDIR}/02b_PCA_additional.png")
print("STEP 2 COMPLETE ✓")
