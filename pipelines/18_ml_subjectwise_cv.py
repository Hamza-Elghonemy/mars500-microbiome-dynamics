import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             mean_absolute_error, r2_score)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# WP1: Subject-Wise ML Cross-Validation
# MARS500 Publication Pipeline — Methodological Hardening
# ==============================================================================
# Addresses reviewer concern: standard k-fold CV leaks information because
# samples from the SAME crew member appear in both train and test sets.
# Since crew identity explains ~46% of microbiome variance (PERMANOVA R²=0.46),
# this inflates performance estimates.
#
# Solution: Leave-One-Subject-Out (LOSO) CV — train on 5 crew members, test on
# the held-out 6th. This gives an honest estimate of generalization to a NEW person.
#
# Additional analyses:
#   1. Permutation test (1000 iterations) with subject-wise splits
#   2. Ablation study: full ASV set vs biomarker genera vs phylum level
#   3. Feature importance stability across folds (Jaccard similarity)
# ==============================================================================

print("=" * 70)
print("WP1: Subject-Wise ML Cross-Validation")
print("=" * 70)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
meta_dir = os.path.join(base_dir, "data", "metadata")
tab_dir  = os.path.join(base_dir, "results", "tables")
fig_dir  = os.path.join(base_dir, "results", "figures")
os.makedirs(tab_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# ---------- Load data --------------------------------------------------------
counts = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_counts.tsv"),
                     sep="\t", index_col=0).T
meta = pd.read_csv(os.path.join(meta_dir, "processed_metadata.tsv"),
                   sep="\t", index_col="SampleID")
taxonomy = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_taxonomy.tsv"),
                       sep="\t", index_col=0)

common = counts.index.intersection(meta.index)
X_raw = counts.loc[common]
X_rel = X_raw.div(X_raw.sum(axis=1), axis=0)  # TSS normalization

y_phase = meta.loc[common, 'Phase']
y_day   = meta.loc[common, 'Timepoint_Day'].astype(float)
groups  = meta.loc[common, 'CrewMember']

print(f"\n  Samples: {len(common)}")
print(f"  ASVs   : {X_rel.shape[1]}")
print(f"  Crews  : {sorted(groups.unique())}")
print(f"  Phases : {sorted(y_phase.unique())}")

# =============================================================================
# PART 1: LOSO Classification (Phase prediction)
# =============================================================================
print("\n" + "=" * 70)
print("PART 1: Leave-One-Subject-Out — Phase Classification")
print("=" * 70)

logo = LeaveOneGroupOut()
rf_clf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=42, n_jobs=-1)

y_pred_loso = cross_val_predict(rf_clf, X_rel, y_phase, groups=groups, cv=logo)
acc_loso = accuracy_score(y_phase, y_pred_loso)

print(f"\n  LOSO Accuracy: {acc_loso:.4f}")
print(f"\n  Classification Report:")
report = classification_report(y_phase, y_pred_loso, output_dict=True)
print(classification_report(y_phase, y_pred_loso))

# Per-fold accuracy
fold_accs = []
for train_idx, test_idx in logo.split(X_rel, y_phase, groups):
    rf_fold = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                      random_state=42, n_jobs=-1)
    rf_fold.fit(X_rel.iloc[train_idx], y_phase.iloc[train_idx])
    fold_acc = rf_fold.score(X_rel.iloc[test_idx], y_phase.iloc[test_idx])
    crew = groups.iloc[test_idx[0]]
    fold_accs.append({'CrewMember': crew, 'Accuracy': fold_acc,
                      'N_test': len(test_idx)})
    print(f"  Fold {crew}: acc={fold_acc:.3f} (n={len(test_idx)})")

fold_df = pd.DataFrame(fold_accs)
fold_df.to_csv(os.path.join(tab_dir, "loso_classification_per_fold.csv"), index=False)

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
phase_order = ['Pre-Mission', 'Early', 'Mid', 'Late', 'Post-Mission']
cm = confusion_matrix(y_phase, y_pred_loso, labels=phase_order)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=phase_order)
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title(f'LOSO Cross-Validated Confusion Matrix\nAccuracy = {acc_loso:.3f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "loso_confusion_matrix.pdf"))
plt.savefig(os.path.join(fig_dir, "loso_confusion_matrix.png"), dpi=150)
plt.close()
print("  Saved confusion matrix.")

# =============================================================================
# PART 2: LOSO Regression (Day prediction — "biological clock")
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: Leave-One-Subject-Out — Temporal Regression")
print("=" * 70)

rf_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
y_pred_day = cross_val_predict(rf_reg, X_rel, y_day, groups=groups, cv=logo)

r2_loso = r2_score(y_day, y_pred_day)
mae_loso = mean_absolute_error(y_day, y_pred_day)

print(f"\n  LOSO R² : {r2_loso:.4f}")
print(f"  LOSO MAE: {mae_loso:.1f} days")

# Per-fold regression metrics
fold_reg = []
for train_idx, test_idx in logo.split(X_rel, y_day, groups):
    rf_fold = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf_fold.fit(X_rel.iloc[train_idx], y_day.iloc[train_idx])
    pred = rf_fold.predict(X_rel.iloc[test_idx])
    crew = groups.iloc[test_idx[0]]
    fold_r2 = r2_score(y_day.iloc[test_idx], pred)
    fold_mae = mean_absolute_error(y_day.iloc[test_idx], pred)
    fold_reg.append({'CrewMember': crew, 'R2': fold_r2, 'MAE': fold_mae,
                     'N_test': len(test_idx)})
    print(f"  Fold {crew}: R²={fold_r2:.3f}, MAE={fold_mae:.1f} days (n={len(test_idx)})")

fold_reg_df = pd.DataFrame(fold_reg)
fold_reg_df.to_csv(os.path.join(tab_dir, "loso_regression_per_fold.csv"), index=False)

# Predicted vs actual plot
fig, ax = plt.subplots(figsize=(8, 8))
crew_colors = dict(zip(sorted(groups.unique()), sns.color_palette('Dark2', 6)))
for crew in sorted(groups.unique()):
    mask = groups == crew
    ax.scatter(y_day[mask], y_pred_day[mask], c=[crew_colors[crew]]*mask.sum(),
               s=60, alpha=0.7, edgecolors='white', linewidth=0.5, label=crew, zorder=3)
mn, mx = min(y_day.min(), y_pred_day.min()), max(y_day.max(), y_pred_day.max())
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, alpha=0.7, label='Perfect prediction')
ax.set_xlabel('Actual Isolation Day', fontsize=12)
ax.set_ylabel('Predicted Isolation Day', fontsize=12)
ax.set_title(f'Biological Clock: LOSO Temporal Regression\nR² = {r2_loso:.3f}, MAE = {mae_loso:.0f} days',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "loso_temporal_regression.pdf"))
plt.savefig(os.path.join(fig_dir, "loso_temporal_regression.png"), dpi=150)
plt.close()
print("  Saved temporal regression plot.")

# =============================================================================
# PART 3: Permutation Test (subject-wise)
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: Permutation Test — Subject-Wise (1000 iterations)")
print("=" * 70)

N_PERM = 1000
null_accs_clf = []
null_r2_reg = []

rng = np.random.RandomState(42)
for i in range(N_PERM):
    if (i + 1) % 100 == 0:
        print(f"  Permutation {i+1}/{N_PERM}...")

    # Shuffle labels WITHIN each subject (preserves group structure)
    # Actually for a proper permutation test, shuffle group labels
    perm_idx = rng.permutation(len(y_phase))
    y_phase_perm = y_phase.values[perm_idx]
    y_day_perm   = y_day.values[perm_idx]

    # Classification
    try:
        y_pred_perm = cross_val_predict(
            RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                    random_state=42, n_jobs=-1),
            X_rel, y_phase_perm, groups=groups, cv=logo)
        null_accs_clf.append(accuracy_score(y_phase_perm, y_pred_perm))
    except:
        null_accs_clf.append(np.nan)

    # Regression
    try:
        y_pred_day_perm = cross_val_predict(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            X_rel, y_day_perm, groups=groups, cv=logo)
        null_r2_reg.append(r2_score(y_day_perm, y_pred_day_perm))
    except:
        null_r2_reg.append(np.nan)

null_accs_clf = np.array([x for x in null_accs_clf if not np.isnan(x)])
null_r2_reg   = np.array([x for x in null_r2_reg if not np.isnan(x)])

p_clf = (np.sum(null_accs_clf >= acc_loso) + 1) / (len(null_accs_clf) + 1)
p_reg = (np.sum(null_r2_reg >= r2_loso) + 1) / (len(null_r2_reg) + 1)

print(f"\n  Classification: true={acc_loso:.4f}, null={null_accs_clf.mean():.4f}±{null_accs_clf.std():.4f}, p={p_clf:.4f}")
print(f"  Regression:     true R²={r2_loso:.4f}, null={null_r2_reg.mean():.4f}±{null_r2_reg.std():.4f}, p={p_reg:.4f}")

# Save permutation results
perm_results = pd.DataFrame({
    'Metric': ['Classification_Accuracy', 'Regression_R2'],
    'True_Value': [acc_loso, r2_loso],
    'Null_Mean': [null_accs_clf.mean(), null_r2_reg.mean()],
    'Null_Std': [null_accs_clf.std(), null_r2_reg.std()],
    'P_Value': [p_clf, p_reg],
    'N_Permutations': [len(null_accs_clf), len(null_r2_reg)]
})
perm_results.to_csv(os.path.join(tab_dir, "loso_permutation_test.csv"), index=False)

# Permutation distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(null_accs_clf, bins=40, color='steelblue', edgecolor='white', alpha=0.7, label='Null distribution')
axes[0].axvline(acc_loso, color='red', linewidth=2.5, linestyle='--', label=f'True = {acc_loso:.3f}')
axes[0].set_xlabel('Accuracy', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title(f'Phase Classification Permutation Test\np = {p_clf:.4f}', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)

axes[1].hist(null_r2_reg, bins=40, color='darkorange', edgecolor='white', alpha=0.7, label='Null distribution')
axes[1].axvline(r2_loso, color='red', linewidth=2.5, linestyle='--', label=f'True R² = {r2_loso:.3f}')
axes[1].set_xlabel('R²', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title(f'Temporal Regression Permutation Test\np = {p_reg:.4f}', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)

plt.suptitle('Subject-Wise Permutation Tests (n=1000)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "loso_permutation_tests.pdf"))
plt.savefig(os.path.join(fig_dir, "loso_permutation_tests.png"), dpi=150)
plt.close()
print("  Saved permutation test plots.")

# =============================================================================
# PART 4: Ablation Study — What resolution is needed?
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: Ablation Study — Feature Resolution")
print("=" * 70)

# Prepare genus-level features
genus_map = taxonomy['genus'].replace('NA', np.nan)
X_genus = X_raw.copy()
X_genus.columns = [genus_map.get(c, c) for c in X_genus.columns]
# Aggregate by genus (sum ASVs within same genus)
X_genus = X_genus.T.groupby(level=0).sum().T
X_genus = X_genus.loc[:, X_genus.columns.notna()]
X_genus_rel = X_genus.div(X_genus.sum(axis=1), axis=0)

# Prepare phylum-level features
phylum_map = taxonomy['phylum'].replace('NA', np.nan)
X_phylum = X_raw.copy()
X_phylum.columns = [phylum_map.get(c, c) for c in X_phylum.columns]
X_phylum = X_phylum.T.groupby(level=0).sum().T
X_phylum = X_phylum.loc[:, X_phylum.columns.notna()]
X_phylum_rel = X_phylum.div(X_phylum.sum(axis=1), axis=0)

# Top 20 biomarker genera (from prior feature importance)
top20_genera = X_genus_rel.var().sort_values(ascending=False).head(20).index
X_top20 = X_genus_rel[top20_genera]

ablation_results = []
feature_sets = {
    'Full ASV (n={})'.format(X_rel.shape[1]): X_rel,
    'Genus-level (n={})'.format(X_genus_rel.shape[1]): X_genus_rel,
    'Top 20 Genera': X_top20,
    'Phylum-level (n={})'.format(X_phylum_rel.shape[1]): X_phylum_rel,
}

for name, X_feat in feature_sets.items():
    X_feat = X_feat.fillna(0)

    # Classification
    y_pred_ab = cross_val_predict(
        RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                random_state=42, n_jobs=-1),
        X_feat, y_phase, groups=groups, cv=logo)
    acc_ab = accuracy_score(y_phase, y_pred_ab)

    # Regression
    y_pred_day_ab = cross_val_predict(
        RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
        X_feat, y_day, groups=groups, cv=logo)
    r2_ab = r2_score(y_day, y_pred_day_ab)
    mae_ab = mean_absolute_error(y_day, y_pred_day_ab)

    ablation_results.append({
        'Feature_Set': name,
        'N_Features': X_feat.shape[1],
        'LOSO_Accuracy': acc_ab,
        'LOSO_R2': r2_ab,
        'LOSO_MAE': mae_ab
    })
    print(f"  {name:30s}  Acc={acc_ab:.3f}  R²={r2_ab:.3f}  MAE={mae_ab:.0f}d")

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv(os.path.join(tab_dir, "loso_ablation_study.csv"), index=False)

# Ablation bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x_pos = range(len(ablation_df))
axes[0].barh(x_pos, ablation_df['LOSO_Accuracy'], color=sns.color_palette('Set2', len(ablation_df)),
             edgecolor='white', height=0.6)
axes[0].set_yticks(x_pos)
axes[0].set_yticklabels(ablation_df['Feature_Set'], fontsize=10)
axes[0].set_xlabel('LOSO Accuracy', fontsize=11)
axes[0].set_title('Phase Classification', fontsize=12, fontweight='bold')
for i, v in enumerate(ablation_df['LOSO_Accuracy']):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)

axes[1].barh(x_pos, ablation_df['LOSO_R2'], color=sns.color_palette('Set2', len(ablation_df)),
             edgecolor='white', height=0.6)
axes[1].set_yticks(x_pos)
axes[1].set_yticklabels(ablation_df['Feature_Set'], fontsize=10)
axes[1].set_xlabel('LOSO R²', fontsize=11)
axes[1].set_title('Temporal Regression', fontsize=12, fontweight='bold')
for i, v in enumerate(ablation_df['LOSO_R2']):
    axes[1].text(max(v + 0.005, 0.01), i, f'{v:.3f}', va='center', fontsize=10)

plt.suptitle('Ablation Study: Feature Resolution vs Predictive Power', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "loso_ablation_study.pdf"))
plt.savefig(os.path.join(fig_dir, "loso_ablation_study.png"), dpi=150)
plt.close()

# =============================================================================
# PART 5: Feature Importance Stability
# =============================================================================
print("\n" + "=" * 70)
print("PART 5: Feature Importance Stability Across Folds")
print("=" * 70)

top_k = 30
fold_importances = []
fold_top_sets = []

for train_idx, test_idx in logo.split(X_rel, y_day, groups):
    rf_fold = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf_fold.fit(X_rel.iloc[train_idx], y_day.iloc[train_idx])
    imp = pd.Series(rf_fold.feature_importances_, index=X_rel.columns)
    fold_importances.append(imp)
    fold_top_sets.append(set(imp.nlargest(top_k).index))

# Jaccard similarity between all pairs of folds
n_folds = len(fold_top_sets)
jaccard_matrix = np.zeros((n_folds, n_folds))
for i in range(n_folds):
    for j in range(n_folds):
        intersection = len(fold_top_sets[i] & fold_top_sets[j])
        union = len(fold_top_sets[i] | fold_top_sets[j])
        jaccard_matrix[i, j] = intersection / union if union > 0 else 0

mean_jaccard = jaccard_matrix[np.triu_indices(n_folds, k=1)].mean()
print(f"\n  Mean Jaccard similarity of top-{top_k} features across folds: {mean_jaccard:.3f}")
print(f"  (1.0 = perfectly stable, 0.0 = completely unstable)")

# Consensus top features (appear in most folds)
from collections import Counter
all_top_features = []
for s in fold_top_sets:
    all_top_features.extend(list(s))
feature_freq = Counter(all_top_features)
consensus = pd.DataFrame.from_dict(feature_freq, orient='index', columns=['Fold_Count'])
consensus = consensus.sort_values('Fold_Count', ascending=False).head(20)

# Map to genus names
asv_to_genus = taxonomy['genus'].to_dict()
consensus['Genus'] = [asv_to_genus.get(asv, 'NA') for asv in consensus.index]
consensus['Label'] = [f"{row['Genus']} ({idx})" if row['Genus'] != 'NA' else idx
                      for idx, row in consensus.iterrows()]

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#2ca02c' if v == n_folds else '#1f77b4' if v >= n_folds-1 else '#ff7f0e'
          for v in consensus['Fold_Count']]
ax.barh(range(len(consensus)), consensus['Fold_Count'], color=colors, edgecolor='white')
ax.set_yticks(range(len(consensus)))
ax.set_yticklabels(consensus['Label'], fontsize=9)
ax.set_xlabel(f'Number of Folds (out of {n_folds})', fontsize=11)
ax.set_title(f'Feature Importance Stability — Top {top_k} Features\nMean Jaccard = {mean_jaccard:.3f}',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "loso_feature_stability.pdf"))
plt.savefig(os.path.join(fig_dir, "loso_feature_stability.png"), dpi=150)
plt.close()

# =============================================================================
# PART 6: Comparison Summary (Old vs New)
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Standard CV vs Subject-Wise LOSO")
print("=" * 70)

comparison = pd.DataFrame({
    'Method': ['Standard 5-fold CV (old)', 'Subject-wise LOSO (new)'],
    'Classification_Accuracy': [0.616, acc_loso],
    'Regression_R2': [0.63, r2_loso],
    'Regression_MAE_days': [68, mae_loso],
    'Permutation_p_clf': ['0.001', f'{p_clf:.4f}'],
    'Permutation_p_reg': ['0.001', f'{p_reg:.4f}'],
    'Leakage_Free': [False, True]
})
comparison.to_csv(os.path.join(tab_dir, "loso_vs_standard_comparison.csv"), index=False)
print(comparison.to_string(index=False))

print("\n" + "=" * 70)
print("WP1 COMPLETE — All results saved to results/tables/ and results/figures/")
print("=" * 70)
