"""
==============================================================================
STEP 5: MACHINE LEARNING PROGNOSTIC CLASSIFIER
  - Random Forest classifier for cluster assignment
  - Cross-validated AUC for each cluster (one-vs-rest)
  - Feature importance → prognostic gene panel
  - Logistic regression baseline comparison
  - Concordance index (C-index) for survival prediction
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    label_binarize
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score
)

from sklearn.multiclass import OneVsRestClassifier

# =============================================================================
# PROJECT PATHS
# =============================================================================
BASE_DIR = "/content/drive/MyDrive/TCGA_Project"

OUTDIR = os.path.join(BASE_DIR, "results")
FIGDIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

print("=" * 60)
print("STEP 5: MACHINE LEARNING CLASSIFIER")
print("=" * 60)

# ── Load processed data ────────────────────────────────────────────────────
EXPR_INPUT_PATH = os.path.join(
    OUTDIR,
    "expr_filtered.pkl"
)

CLIN_INPUT_PATH = os.path.join(
    OUTDIR,
    "clin_clustered.pkl"
)

PCA_INPUT_PATH = os.path.join(
    OUTDIR,
    "X_pca.npy"
)

with open(EXPR_INPUT_PATH, 'rb') as f:
    expr = pickle.load(f)

with open(CLIN_INPUT_PATH, 'rb') as f:
    clin = pickle.load(f)

X_pca = np.load(PCA_INPUT_PATH)

print(f"[Loaded] {EXPR_INPUT_PATH}")
print(f"[Loaded] {CLIN_INPUT_PATH}")
print(f"[Loaded] {PCA_INPUT_PATH}")

clusters = sorted(
    clin['cluster_label'].unique()
)

le = LabelEncoder()

y = le.fit_transform(
    clin['cluster_label'].values
)

print(f"Classes: {le.classes_}, n={len(y)}")

# ── Use top DEG genes as features ──────────────────────────────────────────
top_feat_genes = []

for cl in clusters:

    DEG_PATH = os.path.join(
        OUTDIR,
        f"DEG_{cl}.csv"
    )

    deg = pd.read_csv(DEG_PATH)

    sig = deg[
        (deg['p_adj'] < 0.05) &
        (deg['log2FC'].abs() > 0.5)
    ]

    top_feat_genes.extend(
        sig
        .nsmallest(200, 'p_value')['gene']
        .tolist()
    )

top_feat_genes = list(
    dict.fromkeys(top_feat_genes)
)[:500]

top_feat_genes = [
    g for g in top_feat_genes
    if g in expr.columns
]

print(f"Feature set: {len(top_feat_genes)} genes")

X = expr[top_feat_genes].values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ── Cross-validated Random Forest ──────────────────────────────────────────
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

print("\nRunning 5-fold cross-validation (Random Forest)...")

y_pred_proba = cross_val_predict(
    rf,
    X_scaled,
    y,
    cv=cv,
    method='predict_proba'
)

y_pred_class = y_pred_proba.argmax(axis=1)

acc = accuracy_score(
    y,
    y_pred_class
)

print(f"  CV Accuracy: {acc:.4f}")

# ── One-vs-rest AUC ────────────────────────────────────────────────────────
y_bin = label_binarize(
    y,
    classes=range(len(clusters))
)

aucs = {}

for i, cl in enumerate(clusters):

    auc = roc_auc_score(
        y_bin[:, i],
        y_pred_proba[:, i]
    )

    aucs[cl] = auc

    print(f"  {cl} AUC (OvR): {auc:.4f}")

mean_auc = np.mean(
    list(aucs.values())
)

print(f"  Mean AUC: {mean_auc:.4f}")

# ── Logistic Regression baseline ──────────────────────────────────────────
print("\nLogistic Regression baseline (PCA features)...")

lr = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42,
    multi_class='ovr',
    solver='lbfgs'
)

y_lr_proba = cross_val_predict(
    lr,
    X_pca[:, :20],
    y,
    cv=cv,
    method='predict_proba'
)

y_lr_class = y_lr_proba.argmax(axis=1)

lr_acc = accuracy_score(
    y,
    y_lr_class
)

lr_aucs = {}

for i, cl in enumerate(clusters):

    a = roc_auc_score(
        y_bin[:, i],
        y_lr_proba[:, i]
    )

    lr_aucs[cl] = a

print(
    f"  LR Accuracy: {lr_acc:.4f}, "
    f"Mean AUC: {np.mean(list(lr_aucs.values())):.4f}"
)

# ── Feature importance ─────────────────────────────────────────────────────
print("\nFitting final RF on full data for feature importance...")

rf.fit(X_scaled, y)

importances = pd.Series(
    rf.feature_importances_,
    index=top_feat_genes
)

top50_imp = importances.nlargest(50)

print("Top 10 prognostic genes by RF importance:")

for g, v in top50_imp.head(10).items():

    print(f"  {g}: {v:.5f}")

FEATURE_IMPORTANCE_PATH = os.path.join(
    OUTDIR,
    "rf_feature_importance.csv"
)

top50_imp.to_csv(
    FEATURE_IMPORTANCE_PATH,
    header=['importance']
)

print(f"[Saved] {FEATURE_IMPORTANCE_PATH}")

# ── Concordance index (C-index) ────────────────────────────────────────────
high_risk_cl = (
    clusters.index('C3')
    if 'C3' in clusters
    else 0
)

risk_score = y_pred_proba[:, high_risk_cl]

def concordance_index(time, event, risk):
    """
    Harrell's concordance index.
    """

    n = len(time)

    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):

        for j in range(i + 1, n):

            if event[i] == 0 and event[j] == 0:
                continue

            if time[i] == time[j]:
                continue

            if event[i] == 1 and time[i] < time[j]:

                concordant += (risk[i] > risk[j])
                discordant += (risk[i] < risk[j])
                tied += (risk[i] == risk[j])

            elif event[j] == 1 and time[j] < time[i]:

                concordant += (risk[j] > risk[i])
                discordant += (risk[j] < risk[i])
                tied += (risk[i] == risk[j])

    denom = concordant + discordant + tied

    return (
        concordant / denom
        if denom > 0
        else 0.5
    )

# ── Sample for faster computation ──────────────────────────────────────────
np.random.seed(42)

sample_idx = np.random.choice(
    len(risk_score),
    size=min(500, len(risk_score)),
    replace=False
)

cindex = concordance_index(
    clin['OS.time'].values[sample_idx],
    clin['OS'].values[sample_idx],
    risk_score[sample_idx]
)

print(
    f"\nHarrell's C-index "
    f"(OS, n=500 sample): {cindex:.4f}"
)

print(
    "  (0.5=random, >0.7=good, >0.8=excellent)"
)

# ── ROC curve figure ───────────────────────────────────────────────────────
colors = [
    '#1f77b4',
    '#d62728',
    '#2ca02c',
    '#9467bd'
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

fig.suptitle(
    "TCGA-BRCA: ML Classifier Performance",
    fontsize=13,
    fontweight='bold'
)

# ── ROC curves ─────────────────────────────────────────────────────────────
ax = axes[0]

ax.plot(
    [0, 1],
    [0, 1],
    'k--',
    lw=1,
    label='Random'
)

for i, cl in enumerate(clusters):

    fpr, tpr, _ = roc_curve(
        y_bin[:, i],
        y_pred_proba[:, i]
    )

    ax.plot(
        fpr,
        tpr,
        lw=2,
        color=colors[i],
        label=f'{cl} AUC={aucs[cl]:.3f}'
    )

ax.set_xlabel(
    "False Positive Rate",
    fontsize=11
)

ax.set_ylabel(
    "True Positive Rate",
    fontsize=11
)

ax.set_title(
    f"Random Forest ROC\n"
    f"(5-fold CV, mean AUC={mean_auc:.3f})",
    fontsize=11
)

ax.legend(fontsize=10)

ax.grid(alpha=0.3)

# ── Feature importance plot ────────────────────────────────────────────────
ax = axes[1]

top15 = importances.nlargest(15)

colors_bar = ['#2196F3'] * 15

ax.barh(
    range(15),
    top15.values[::-1],
    color=colors_bar,
    edgecolor='white'
)

ax.set_yticks(range(15))

ax.set_yticklabels(
    top15.index[::-1],
    fontsize=10
)

ax.set_xlabel(
    "Feature Importance (Gini)",
    fontsize=11
)

ax.set_title(
    "Top 15 Prognostic Genes\n"
    "(Random Forest Feature Importance)",
    fontsize=11
)

ax.grid(axis='x', alpha=0.3)

plt.tight_layout()

ML_FIG_PATH = os.path.join(
    FIGDIR,
    "05_ML_performance.png"
)

plt.savefig(
    ML_FIG_PATH,
    dpi=150,
    bbox_inches='tight'
)

plt.close()

# ── Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(
    y,
    y_pred_class
)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax
)

ax.set_xlabel(
    "Predicted Cluster",
    fontsize=12
)

ax.set_ylabel(
    "True Cluster",
    fontsize=12
)

ax.set_title(
    f"Confusion Matrix (5-fold CV)\n"
    f"Accuracy={acc:.3f}",
    fontsize=12
)

plt.tight_layout()

CM_FIG_PATH = os.path.join(
    FIGDIR,
    "05b_confusion_matrix.png"
)

plt.savefig(
    CM_FIG_PATH,
    dpi=150,
    bbox_inches='tight'
)

plt.close()

# ── Save performance metrics ───────────────────────────────────────────────
metrics = {
    'model': [
        'RandomForest',
        'LogisticRegression'
    ],
    'accuracy': [
        round(acc, 4),
        round(lr_acc, 4)
    ],
    'mean_AUC': [
        round(mean_auc, 4),
        round(np.mean(list(lr_aucs.values())), 4)
    ],
    'C_index_OS': [
        round(cindex, 4),
        'NA'
    ],
}

ML_METRICS_PATH = os.path.join(
    OUTDIR,
    "ml_performance.csv"
)

pd.DataFrame(metrics).to_csv(
    ML_METRICS_PATH,
    index=False
)

print(f"\n[Saved] {ML_FIG_PATH}")
print(f"[Saved] {CM_FIG_PATH}")
print(f"[Saved] {ML_METRICS_PATH}")

print("STEP 5 COMPLETE ✓")
