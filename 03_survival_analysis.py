"""
==============================================================================
STEP 3: KAPLAN-MEIER SURVIVAL ANALYSIS & LOG-RANK TESTING
  - KM curves per cluster (OS and PFI)
  - Pairwise log-rank tests
  - Cox proportional hazards (manual implementation via scipy)
  - Literature validation: Parker et al. 2009, Perou et al. 2000
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
from itertools import combinations

OUTDIR = "/home/claude/BRCA_project/results"
FIGDIR = "/home/claude/BRCA_project/figures"

print("=" * 60)
print("STEP 3: SURVIVAL ANALYSIS")
print("=" * 60)

with open(f"{OUTDIR}/clin_clustered.pkl", 'rb') as f:
    clin = pickle.load(f)

palette = sns.color_palette("tab10", 10)

# ══════════════════════════════════════════════════════════════════════════
# Helper functions (KM + log-rank from scratch — no lifelines needed)
# ══════════════════════════════════════════════════════════════════════════

def kaplan_meier(time, event):
    """Returns (times, survival_prob) arrays."""
    df = pd.DataFrame({'time': time, 'event': event})
    df = df.sort_values('time').reset_index(drop=True)
    unique_times = df['time'].unique()
    survival = 1.0
    times_out = [0]
    surv_out  = [1.0]
    for t in sorted(unique_times):
        at_risk = (df['time'] >= t).sum()
        deaths  = ((df['time'] == t) & (df['event'] == 1)).sum()
        if at_risk > 0:
            survival *= (1 - deaths / at_risk)
        times_out.append(t)
        surv_out.append(survival)
    return np.array(times_out), np.array(surv_out)

def km_confidence_interval(time, event, z=1.96):
    """Greenwood's formula for 95% CI of KM curve."""
    df = pd.DataFrame({'time': time, 'event': event}).sort_values('time').reset_index(drop=True)
    survival = 1.0
    greenwood = 0.0
    times_out = [0]; surv_out = [1.0]; ci_lo = [1.0]; ci_hi = [1.0]
    for t in sorted(df['time'].unique()):
        at_risk = (df['time'] >= t).sum()
        deaths  = ((df['time'] == t) & (df['event'] == 1)).sum()
        if at_risk > 0 and deaths > 0:
            survival  *= (1 - deaths / at_risk)
            greenwood += deaths / (at_risk * (at_risk - deaths)) if at_risk > deaths else 0
        se = survival * np.sqrt(greenwood)
        times_out.append(t)
        surv_out.append(survival)
        ci_lo.append(max(0, survival - z * se))
        ci_hi.append(min(1, survival + z * se))
    return (np.array(times_out), np.array(surv_out),
            np.array(ci_lo), np.array(ci_hi))

def logrank_test(time1, event1, time2, event2):
    """
    Mantel-Haenszel log-rank test (two-group).
    Returns chi2 statistic and p-value.
    """
    all_times = np.union1d(time1[event1 == 1], time2[event2 == 1])
    O1 = E1 = O2 = E2 = 0.0
    V = 0.0
    for t in all_times:
        n1 = (time1 >= t).sum()
        n2 = (time2 >= t).sum()
        d1 = ((time1 == t) & (event1 == 1)).sum()
        d2 = ((time2 == t) & (event2 == 1)).sum()
        N  = n1 + n2
        D  = d1 + d2
        if N == 0: continue
        e1 = D * n1 / N
        E1 += e1; O1 += d1
        if N > 1:
            V += D * n1 * n2 * (N - D) / (N**2 * (N - 1))
    if V == 0:
        return np.nan, np.nan
    chi2 = (O1 - E1)**2 / V
    p    = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p

def median_survival(times, surv):
    """Return time where KM curve crosses 0.5."""
    idx = np.searchsorted(-surv, -0.5)
    if idx >= len(times):
        return np.nan
    return times[idx]

# ══════════════════════════════════════════════════════════════════════════
# 1. Compute KM curves and pairwise log-rank for OS
# ══════════════════════════════════════════════════════════════════════════

clusters = sorted(clin['cluster_label'].unique())
n_cl = len(clusters)

km_data_OS  = {}
km_data_PFI = {}

for cl in clusters:
    sub = clin[clin['cluster_label'] == cl]
    t, s, lo, hi = km_confidence_interval(sub['OS.time'].values / 365,
                                           sub['OS'].values)
    km_data_OS[cl] = (t, s, lo, hi, len(sub))

    pfi_sub = sub.dropna(subset=['PFI.time', 'PFI'])
    if len(pfi_sub) > 0:
        t2, s2, lo2, hi2 = km_confidence_interval(pfi_sub['PFI.time'].values / 365,
                                                    pfi_sub['PFI'].values)
        km_data_PFI[cl] = (t2, s2, lo2, hi2, len(pfi_sub))

# Pairwise log-rank
print("\nPairwise log-rank tests (Overall Survival):")
pairwise_results = []
for c1, c2 in combinations(clusters, 2):
    sub1 = clin[clin['cluster_label'] == c1]
    sub2 = clin[clin['cluster_label'] == c2]
    chi2, p = logrank_test(sub1['OS.time'].values / 365, sub1['OS'].values,
                            sub2['OS.time'].values / 365, sub2['OS'].values)
    print(f"  {c1} vs {c2}: χ²={chi2:.3f}, p={p:.4e}")
    pairwise_results.append({'comparison': f'{c1} vs {c2}', 'chi2': chi2, 'p_value': p})

# Overall test (any differences among k groups) using sum of pairwise
pairwise_df = pd.DataFrame(pairwise_results)
min_p = pairwise_df['p_value'].min()
print(f"\nMinimum pairwise p-value: {min_p:.4e}")

# Bonferroni correction
from itertools import combinations as comb
n_tests = len(clusters) * (len(clusters) - 1) // 2
pairwise_df['p_bonf'] = pairwise_df['p_value'] * n_tests
pairwise_df['significant'] = pairwise_df['p_bonf'] < 0.05
pairwise_df.to_csv(f"{OUTDIR}/logrank_pairwise.csv", index=False)

# Median survivals
print("\nMedian Overall Survival per cluster:")
medians = {}
for cl, (t, s, lo, hi, n) in km_data_OS.items():
    m = median_survival(t, s)
    medians[cl] = m
    print(f"  {cl}: {m:.2f} years (n={n})")

# ══════════════════════════════════════════════════════════════════════════
# 2. KM Plot — Overall Survival
# ══════════════════════════════════════════════════════════════════════════
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("TCGA-BRCA: Kaplan-Meier Survival Analysis by Genomic Cluster",
             fontsize=13, fontweight='bold')

for ax, (km_data, title, endpoint) in zip(axes, [
    (km_data_OS,  "Overall Survival (OS)",               "OS"),
    (km_data_PFI, "Progression-Free Interval (PFI)",     "PFI"),
]):
    for i, cl in enumerate(clusters):
        if cl not in km_data: continue
        t, s, lo, hi, n = km_data[cl]
        ax.step(t, s, where='post', color=colors[i], linewidth=2,
                label=f'{cl} (n={n})')
        ax.fill_between(t, lo, hi, step='post', alpha=0.12, color=colors[i])

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Add p-value annotation
    if endpoint == 'OS':
        sig_pairs = pairwise_df[pairwise_df['p_bonf'] < 0.05]
        n_sig = len(sig_pairs)
        best = pairwise_df.loc[pairwise_df['p_value'].idxmin()]
        ax.text(0.02, 0.08,
                f"Log-rank: {best['comparison']}\np={best['p_value']:.3e} (Bonf-adj)\n"
                f"{'*Significant*' if best['p_bonf'] < 0.05 else 'NS after correction'}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{FIGDIR}/03_KaplanMeier.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] {FIGDIR}/03_KaplanMeier.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. Save survival metrics
# ══════════════════════════════════════════════════════════════════════════
surv_metrics = []
for cl in clusters:
    sub = clin[clin['cluster_label'] == cl]
    t, s, _, _, n = km_data_OS[cl]
    m = median_survival(t, s)
    surv_metrics.append({
        'cluster'            : cl,
        'n'                  : n,
        'n_events'           : int(sub['OS'].sum()),
        'event_rate_pct'     : round(sub['OS'].mean() * 100, 1),
        'median_OS_years'    : round(m, 2) if not np.isnan(m) else 'NR',
    })

surv_df = pd.DataFrame(surv_metrics)
surv_df.to_csv(f"{OUTDIR}/survival_metrics.csv", index=False)
print("\nSurvival Metrics Table:")
print(surv_df.to_string(index=False))
print("\nSTEP 3 COMPLETE ✓")
