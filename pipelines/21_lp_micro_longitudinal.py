import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
warnings.filterwarnings('ignore')

# ==============================================================================
# Pipeline 21: LP-Micro-Inspired Longitudinal Feature Selection & Classification
# MARS500 Temporal Gut Microbiome Dynamics
# ==============================================================================
# Implements an LP-Micro-style framework:
#   1. Polynomial trajectory features per subject × taxon
#   2. Time-windowed abundance features (5 mission-phase windows)
#   3. Group lasso (proximal gradient) to select entire taxon trajectories
#   4. XGBoost classifier with LOSO CV on selected features
#   5. Permutation importance → "taxon × time-window" importance maps
# ==============================================================================

print("=" * 70)
print("Pipeline 21: LP-Micro Longitudinal Feature Selection")
print("=" * 70)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
meta_dir = os.path.join(base_dir, "data", "metadata")
tab_dir  = os.path.join(base_dir, "results", "tables")
fig_dir  = os.path.join(base_dir, "results", "figures")
os.makedirs(tab_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# =============================================================================
# SECTION 1: DATA LOADING & GENUS-LEVEL CLR
# =============================================================================
print("\n[1/6] Loading data...")
counts = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_counts.tsv"),
                     sep="\t", index_col=0)
meta = pd.read_csv(os.path.join(meta_dir, "processed_metadata.tsv"),
                   sep="\t", index_col="SampleID")
taxonomy = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_taxonomy.tsv"),
                       sep="\t", index_col=0)

# Aggregate to genus level
genus_map = taxonomy['genus'].fillna('Unknown')
counts_g = counts.copy()
counts_g.index = [genus_map.get(a, 'Unknown') for a in counts_g.index]
counts_g = counts_g.groupby(level=0).sum()
counts_g = counts_g[counts_g.index != 'Unknown']

# Sanitize genus names (XGBoost 3.x forbids [, ], < in feature names)
import re
def sanitize(name):
    return re.sub(r'[\[\]<>]', '_', str(name))
counts_g.index = [sanitize(g) for g in counts_g.index]
counts_g = counts_g.groupby(level=0).sum()  # re-merge any duplicates from sanitizing

# Align samples
common = [s for s in meta.index if s in counts_g.columns]
counts_g = counts_g[common]
meta = meta.loc[common]

# CLR transform
def clr(df):
    d = df.values.astype(float) + 0.5
    log_d = np.log(d)
    gm = log_d.mean(axis=0, keepdims=True)
    return pd.DataFrame(log_d - gm, index=df.index, columns=df.columns)

clr_g = clr(counts_g)  # genera × samples

# Top 50 most variable genera
var_order = clr_g.var(axis=1).sort_values(ascending=False)
top_genera = var_order.head(50).index.tolist()
clr_top = clr_g.loc[top_genera]

y_phase = meta['Phase']
y_day   = meta['Timepoint_Day'].astype(float)
groups  = meta['CrewMember']

phase_order = ['Pre-Mission', 'Early', 'Mid', 'Late', 'Post-Mission']
print(f"  Samples: {len(common)}, Genera: {len(top_genera)}, Subjects: {groups.nunique()}")

# =============================================================================
# SECTION 2: LONGITUDINAL FEATURE ENGINEERING
# =============================================================================
print("\n[2/6] Engineering longitudinal features...")

# Time windows aligned with mission phases
WINDOWS = {
    'W1_Pre':  (-100, 0),
    'W2_Early': (1, 45),
    'W3_Mid':   (46, 340),
    'W4_Late':  (341, 520),
    'W5_Post':  (521, 800),
}
POLY_DEG = 3

feature_names = []
feature_groups = []  # integer group ID per feature (one per genus)
feature_window_labels = []  # which time window each feature belongs to

# Build feature names/group map
for gi, genus in enumerate(top_genera):
    # Polynomial coefficients (global trajectory shape)
    for d in range(POLY_DEG + 1):
        feature_names.append(f"{genus}__poly_{d}")
        feature_groups.append(gi)
        feature_window_labels.append('trajectory')
    # Time-windowed means
    for wname in WINDOWS:
        feature_names.append(f"{genus}__{wname}")
        feature_groups.append(gi)
        feature_window_labels.append(wname)

n_feat_per_genus = (POLY_DEG + 1) + len(WINDOWS)
n_features = len(feature_names)
print(f"  Features per genus: {n_feat_per_genus} ({POLY_DEG+1} poly + {len(WINDOWS)} windows)")
print(f"  Total features: {n_features} ({len(top_genera)} genera × {n_feat_per_genus})")

# Build feature matrix
X_lp = np.zeros((len(common), n_features))
subjects = groups.unique()

for subj in subjects:
    subj_mask = groups == subj
    subj_samples = meta.index[subj_mask]
    subj_days = y_day[subj_mask].values
    subj_idx = np.where(subj_mask)[0]

    for gi, genus in enumerate(top_genera):
        # Get this subject's trajectory for this genus
        vals = clr_top.loc[genus, subj_samples].values.astype(float)

        # Fit polynomial to trajectory
        if len(subj_days) > POLY_DEG:
            # Normalize days to [0,1] for numerical stability
            d_min, d_max = subj_days.min(), subj_days.max()
            d_range = max(d_max - d_min, 1)
            d_norm = (subj_days - d_min) / d_range
            poly_coefs = np.polyfit(d_norm, vals, POLY_DEG)
        else:
            poly_coefs = np.zeros(POLY_DEG + 1)

        # Assign polynomial coefficients (same for all samples of this subject)
        col_start = gi * n_feat_per_genus
        for d in range(POLY_DEG + 1):
            X_lp[subj_idx, col_start + d] = poly_coefs[d]

        # Time-windowed means
        for wi, (wname, (t_lo, t_hi)) in enumerate(WINDOWS.items()):
            w_mask = (subj_days >= t_lo) & (subj_days <= t_hi)
            w_mean = vals[w_mask].mean() if w_mask.any() else 0.0
            X_lp[subj_idx, col_start + (POLY_DEG + 1) + wi] = w_mean

X_lp_df = pd.DataFrame(X_lp, index=common, columns=feature_names)
feature_groups = np.array(feature_groups)
feature_window_labels = np.array(feature_window_labels)

print(f"  Feature matrix shape: {X_lp_df.shape}")

# =============================================================================
# SECTION 3: GROUP LASSO FEATURE SELECTION
# =============================================================================
print("\n[3/6] Group lasso feature selection (proximal gradient descent)...")

def group_lasso_multiclass(X, y_enc, n_classes, groups, alpha,
                           max_iter=500, lr=None, tol=1e-5):
    """Group lasso for multinomial logistic regression via proximal GD."""
    n, p = X.shape
    W = np.zeros((p, n_classes))
    if lr is None:
        lr = 1.0 / (np.linalg.norm(X, ord=2) ** 2 / n + 1e-8)
    unique_g = np.unique(groups)
    Y_oh = np.eye(n_classes)[y_enc]

    for it in range(max_iter):
        logits = X @ W
        probs = softmax(logits, axis=1)
        grad = X.T @ (probs - Y_oh) / n

        W_new = W - lr * grad

        # Group soft-thresholding
        for g in unique_g:
            mask = groups == g
            w_g = W_new[mask]
            norm_g = np.linalg.norm(w_g)
            if norm_g > alpha * lr:
                W_new[mask] = w_g * (1 - alpha * lr / norm_g)
            else:
                W_new[mask] = 0.0

        if np.linalg.norm(W_new - W) < tol:
            break
        W = W_new

    return W

# Encode labels
le = LabelEncoder()
le.fit(phase_order)
y_enc = le.transform(y_phase)
n_classes = len(phase_order)

# Standardize features
X_std = (X_lp - X_lp.mean(axis=0)) / (X_lp.std(axis=0) + 1e-10)

# Regularization path to find good alpha
alphas = np.logspace(-4, 1.0, 30)
path_results = []
print("  Running regularization path...")

for alpha in alphas:
    W = group_lasso_multiclass(X_std, y_enc, n_classes, feature_groups, alpha)
    # Count selected groups (groups with any non-zero weight)
    selected = set()
    for g in np.unique(feature_groups):
        if np.linalg.norm(W[feature_groups == g]) > 1e-8:
            selected.add(g)
    path_results.append({
        'alpha': alpha, 'n_selected': len(selected),
        'selected_groups': selected, 'W': W.copy()
    })
    if len(selected) > 0:
        print(f"    α={alpha:.3f} → {len(selected)}/{len(top_genera)} genera selected")

# Select alpha that keeps 10-25 genera (sweet spot for interpretability)
best = None
for r in path_results:
    if 5 <= r['n_selected'] <= 25:
        best = r
        break
if best is None:
    # Fall back to the one closest to 15 selected
    candidates = [r for r in path_results if r['n_selected'] > 0]
    if candidates:
        best = min(candidates, key=lambda r: abs(r['n_selected'] - 15))
    else:
        # Group lasso zeroed everything → fall back to top 20 by variance
        print("  [FALLBACK] Group lasso selected 0 genera; using top 20 by variance.")
        fallback_ids = set(range(20))
        best = {'alpha': 0, 'n_selected': 20, 'selected_groups': fallback_ids,
                'W': np.ones((n_features, n_classes)) * 0.01}

selected_group_ids = best['selected_groups']
selected_genera = [top_genera[g] for g in sorted(selected_group_ids)]
selected_mask = np.isin(feature_groups, list(selected_group_ids))
W_best = best['W']

print(f"\n  Selected α = {best['alpha']:.3f}")
print(f"  Selected genera ({len(selected_genera)}): {', '.join(selected_genera)}")

# Group lasso coefficient magnitudes per genus
genus_coef_norms = {}
for g in sorted(selected_group_ids):
    mask = feature_groups == g
    genus_coef_norms[top_genera[g]] = np.linalg.norm(W_best[mask])

# Save selection results
sel_df = pd.DataFrame({
    'Genus': selected_genera,
    'Coef_Norm': [genus_coef_norms[g] for g in selected_genera]
}).sort_values('Coef_Norm', ascending=False)
sel_df.to_csv(os.path.join(tab_dir, "lpmicro_selected_genera.csv"), index=False)

# =============================================================================
# SECTION 4: XGBOOST LOSO CV
# =============================================================================
print("\n[4/6] XGBoost LOSO cross-validation...")

X_selected = X_lp[:, selected_mask]  # numpy array — avoids XGBoost feature name issues
X_full_np = X_lp  # numpy array

logo = LeaveOneGroupOut()

# --- Full LP features (baseline) ---
xgb_full = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              use_label_encoder=False, eval_metric='mlogloss',
                              random_state=42, verbosity=0)
y_pred_full = cross_val_predict(xgb_full, X_full_np, y_enc, groups=groups, cv=logo)
acc_full = accuracy_score(y_enc, y_pred_full)

# --- Group-lasso-selected features ---
xgb_sel = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              use_label_encoder=False, eval_metric='mlogloss',
                              random_state=42, verbosity=0)
y_pred_sel = cross_val_predict(xgb_sel, X_selected, y_enc, groups=groups, cv=logo)
acc_sel = accuracy_score(y_enc, y_pred_sel)

# --- Per-fold breakdown (selected features) ---
fold_results = []
for train_idx, test_idx in logo.split(X_selected, y_enc, groups):
    xgb_f = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                use_label_encoder=False, eval_metric='mlogloss',
                                random_state=42, verbosity=0)
    xgb_f.fit(X_selected[train_idx], y_enc[train_idx])
    acc_f = xgb_f.score(X_selected[test_idx], y_enc[test_idx])
    crew = groups.iloc[test_idx[0]]
    fold_results.append({'Crew': crew, 'Accuracy': acc_f, 'N_test': len(test_idx)})
    print(f"  Fold {crew}: acc={acc_f:.3f} (n={len(test_idx)})")

print(f"\n  LOSO Accuracy (all LP features):      {acc_full:.4f}")
print(f"  LOSO Accuracy (group-lasso selected): {acc_sel:.4f}")

# --- Permutation test (500 iters) ---
print("\n  Permutation test (500 iterations)...")
N_PERM = 500
rng = np.random.RandomState(42)
null_accs = []
for i in range(N_PERM):
    perm_y = rng.permutation(y_enc)
    try:
        yp = cross_val_predict(
            xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False,
                               eval_metric='mlogloss', random_state=42, verbosity=0),
            X_selected, perm_y, groups=groups, cv=logo)
        null_accs.append(accuracy_score(perm_y, yp))
    except:
        pass
    if (i+1) % 100 == 0:
        print(f"    {i+1}/{N_PERM}")

null_accs = np.array(null_accs)
p_val = (np.sum(null_accs >= acc_sel) + 1) / (len(null_accs) + 1)
print(f"  Permutation p-value: {p_val:.4f}")
print(f"  Null accuracy: {null_accs.mean():.4f} ± {null_accs.std():.4f}")

# Save LOSO results
loso_df = pd.DataFrame({
    'Method': ['All LP features', 'Group-lasso selected', 'Previous LOSO (pipeline 18)'],
    'N_Features': [X_full_np.shape[1], X_selected.shape[1], 2916],
    'N_Genera': [len(top_genera), len(selected_genera), 'all ASVs'],
    'LOSO_Accuracy': [acc_full, acc_sel, 0.472],
    'Permutation_p': ['—', f'{p_val:.4f}', '0.001']
})
loso_df.to_csv(os.path.join(tab_dir, "lpmicro_loso_results.csv"), index=False)

# =============================================================================
# SECTION 5: PERMUTATION IMPORTANCE → TAXON × TIME-WINDOW MAPS
# =============================================================================
print("\n[5/6] Permutation importance maps (taxon × time window)...")

# Fit final model on all data for importance analysis
xgb_final = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                use_label_encoder=False, eval_metric='mlogloss',
                                random_state=42, verbosity=0)
xgb_final.fit(X_selected, y_enc)

# Get selected feature names and their window labels
sel_feat_names = np.array(feature_names)[selected_mask]
sel_window_labels = feature_window_labels[selected_mask]

# Compute LOSO accuracy drop when permuting features for each (taxon, window)
window_names = ['trajectory'] + list(WINDOWS.keys())
importance_matrix = np.zeros((len(selected_genera), len(window_names)))
N_PERM_IMP = 50

for gi, genus in enumerate(selected_genera):
    for wi, wname in enumerate(window_names):
        # Find columns matching this genus AND this window
        col_mask = [(genus in fn) and (wl == wname)
                    for fn, wl in zip(sel_feat_names, sel_window_labels)]
        col_indices = np.where(col_mask)[0]
        if len(col_indices) == 0:
            continue

        drops = []
        for _ in range(N_PERM_IMP):
            X_perm = X_selected.copy()
            for ci in col_indices:
                X_perm[:, ci] = rng.permutation(X_perm[:, ci])
            # Use LOSO prediction with permuted features
            acc_perm_vals = []
            for train_idx, test_idx in logo.split(X_perm, y_enc, groups):
                xgb_imp = xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                              use_label_encoder=False,
                                              eval_metric='mlogloss',
                                              random_state=42, verbosity=0)
                xgb_imp.fit(X_perm[train_idx], y_enc[train_idx])
                acc_perm_vals.append(xgb_imp.score(X_perm[test_idx], y_enc[test_idx]))
            drops.append(acc_sel - np.mean(acc_perm_vals))
        importance_matrix[gi, wi] = np.mean(drops)

    if (gi + 1) % 5 == 0:
        print(f"    {gi+1}/{len(selected_genera)} genera processed")

imp_df = pd.DataFrame(importance_matrix, index=selected_genera, columns=window_names)
imp_df.to_csv(os.path.join(tab_dir, "lpmicro_taxon_window_importance.csv"))

# Taxon-level importance (sum across windows)
taxon_importance = imp_df.sum(axis=1).sort_values(ascending=False)
taxon_imp_df = pd.DataFrame({'Genus': taxon_importance.index,
                              'Total_Importance': taxon_importance.values})
taxon_imp_df.to_csv(os.path.join(tab_dir, "lpmicro_taxon_importance.csv"), index=False)

# =============================================================================
# SECTION 6: VISUALIZATION
# =============================================================================
print("\n[6/6] Generating figures...")

# --- Fig 1: Group Lasso Selection Path ---
fig, ax = plt.subplots(figsize=(8, 5))
path_alphas = [r['alpha'] for r in path_results]
path_counts = [r['n_selected'] for r in path_results]
ax.plot(path_alphas, path_counts, 'o-', color='#2C7BB6', linewidth=2, markersize=6)
ax.axhline(len(selected_genera), color='red', linestyle='--', alpha=0.7,
           label=f'Selected: {len(selected_genera)} genera (α={best["alpha"]:.2f})')
ax.set_xscale('log')
ax.set_xlabel('Regularization Strength (α)', fontsize=12)
ax.set_ylabel('Number of Selected Genera', fontsize=12)
ax.set_title('Group Lasso Regularization Path\n(Polynomial Trajectory Features)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "lpmicro_glasso_path.png"), dpi=150)
plt.savefig(os.path.join(fig_dir, "lpmicro_glasso_path.pdf"))
plt.close()

# --- Fig 2: Selected Genera Coefficient Norms ---
fig, ax = plt.subplots(figsize=(8, 6))
sel_sorted = sel_df.sort_values('Coef_Norm', ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sel_sorted)))
ax.barh(range(len(sel_sorted)), sel_sorted['Coef_Norm'], color=colors, edgecolor='white')
ax.set_yticks(range(len(sel_sorted)))
ax.set_yticklabels(sel_sorted['Genus'], fontsize=9)
ax.set_xlabel('Group Lasso Coefficient Norm', fontsize=11)
ax.set_title('LP-Micro: Group-Lasso-Selected Genera\n(Polynomial Trajectory Features)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "lpmicro_selected_genera.png"), dpi=150)
plt.savefig(os.path.join(fig_dir, "lpmicro_selected_genera.pdf"))
plt.close()

# --- Fig 3: LOSO Confusion Matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_enc, y_pred_sel)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=phase_order)
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title(f'LP-Micro LOSO Confusion Matrix\nXGBoost on Group-Lasso Features | Acc = {acc_sel:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "lpmicro_confusion_matrix.png"), dpi=150)
plt.savefig(os.path.join(fig_dir, "lpmicro_confusion_matrix.pdf"))
plt.close()

# --- Fig 4: Permutation Test Distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(null_accs, bins=30, color='steelblue', edgecolor='white', alpha=0.7, label='Null')
ax.axvline(acc_sel, color='red', linewidth=2.5, linestyle='--',
           label=f'True = {acc_sel:.3f}')
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'LP-Micro Permutation Test (p = {p_val:.4f}, n = {N_PERM})',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "lpmicro_permutation_test.png"), dpi=150)
plt.savefig(os.path.join(fig_dir, "lpmicro_permutation_test.pdf"))
plt.close()

# --- Fig 5: TAXON × TIME-WINDOW IMPORTANCE HEATMAP (the key figure) ---
# Sort genera by total importance
imp_sorted = imp_df.loc[taxon_importance.index]
# Rename columns for readability
col_labels = {'trajectory': 'Polynomial\nTrajectory',
              'W1_Pre': 'Pre-Mission\n(≤0d)', 'W2_Early': 'Early\n(1-45d)',
              'W3_Mid': 'Mid\n(46-340d)', 'W4_Late': 'Late\n(341-520d)',
              'W5_Post': 'Post-Mission\n(>520d)'}
imp_plot = imp_sorted.rename(columns=col_labels)

fig, ax = plt.subplots(figsize=(10, max(6, len(selected_genera) * 0.35)))
vmax = max(abs(imp_plot.values.max()), abs(imp_plot.values.min()), 0.01)
sns.heatmap(imp_plot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            vmin=-vmax, vmax=vmax, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Accuracy Drop When Permuted'})
ax.set_title('LP-Micro: Taxon × Time-Window Importance Map\n'
             '"Which taxon at which time window matters for phase prediction"',
             fontsize=13, fontweight='bold')
ax.set_ylabel('')
ax.set_xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "lpmicro_taxon_window_heatmap.png"), dpi=150)
plt.savefig(os.path.join(fig_dir, "lpmicro_taxon_window_heatmap.pdf"))
plt.close()

# --- Fig 6: Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(8, 4))
methods = ['Standard 5-fold CV\n(Pipeline 03)', 'LOSO Raw ASV\n(Pipeline 18)',
           'LP-Micro All Features\n(This Pipeline)', 'LP-Micro Selected\n(This Pipeline)']
accs = [0.616, 0.472, acc_full, acc_sel]
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars = ax.barh(range(len(methods)), accs, color=colors, edgecolor='white', height=0.6)
ax.axvline(0.355, color='gray', linestyle='--', alpha=0.7, label='Chance (~35.5%)')
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel('LOSO Accuracy', fontsize=11)
ax.set_title('Method Comparison: Phase Classification Accuracy',
             fontsize=13, fontweight='bold')
for i, v in enumerate(accs):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "lpmicro_comparison.png"), dpi=150)
plt.savefig(os.path.join(fig_dir, "lpmicro_comparison.pdf"))
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE 21 SUMMARY")
print("=" * 70)
print(f"\n  Feature Engineering:")
print(f"    Top genera:        {len(top_genera)}")
print(f"    Polynomial degree: {POLY_DEG}")
print(f"    Time windows:      {len(WINDOWS)}")
print(f"    Total features:    {n_features}")
print(f"\n  Group Lasso Selection:")
print(f"    α = {best['alpha']:.3f}")
print(f"    Selected genera:   {len(selected_genera)}")
print(f"    Selected features: {selected_mask.sum()}")
print(f"\n  LOSO Classification (XGBoost):")
print(f"    All LP features:      {acc_full:.4f}")
print(f"    Group-lasso selected: {acc_sel:.4f}")
print(f"    Permutation p-value:  {p_val:.4f}")
print(f"    Null mean accuracy:   {null_accs.mean():.4f}")
print(f"\n  Top informative genera (by permutation importance):")
for i, (genus, imp) in enumerate(taxon_importance.head(10).items()):
    print(f"    {i+1}. {genus:25s} Δacc = {imp:+.4f}")
print(f"\n  Figures saved to: {fig_dir}/lpmicro_*.png")
print(f"  Tables saved to:  {tab_dir}/lpmicro_*.csv")
print("=" * 70)
