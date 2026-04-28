import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, r2_score, mean_absolute_error,
                             classification_report, confusion_matrix)

# Models — Classification
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Models — Regression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge,
                                  HuberRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression

# ==============================================================================
# WP1b: Multi-Model LOSO Benchmark
# ==============================================================================
# Tests whether the poor LOSO temporal regression is due to Random Forest
# specifically, or is a fundamental limitation of the data (n=6 subjects).
#
# Model families tested:
#   - Tree ensembles: RF, ExtraTrees, GradientBoosting, XGBoost, AdaBoost
#   - Linear models: Logistic/Ridge Regression, Lasso, ElasticNet, BayesianRidge
#   - Kernel methods: SVM (RBF + linear)
#   - Instance-based: KNN
#   - Neural: MLP (2-layer)
#   - Latent variable: PLS Regression
#
# Feature transformations tested:
#   - TSS (total sum scaling / relative abundance) — standard
#   - CLR (centered log-ratio) — compositionally correct
#   - PCA (20 components on CLR) — dimensionality reduction
# ==============================================================================

print("=" * 70)
print("WP1b: Multi-Model LOSO Benchmark")
print("=" * 70)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
meta_dir = os.path.join(base_dir, "data", "metadata")
tab_dir  = os.path.join(base_dir, "results", "tables")
fig_dir  = os.path.join(base_dir, "results", "figures")

# ---------- Load data --------------------------------------------------------
counts = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_counts.tsv"),
                     sep="\t", index_col=0).T
meta = pd.read_csv(os.path.join(meta_dir, "processed_metadata.tsv"),
                   sep="\t", index_col="SampleID")
taxonomy = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_taxonomy.tsv"),
                       sep="\t", index_col=0)

common = counts.index.intersection(meta.index)
X_raw = counts.loc[common]

y_phase = meta.loc[common, 'Phase']
y_day   = meta.loc[common, 'Timepoint_Day'].astype(float)
groups  = meta.loc[common, 'CrewMember']

logo = LeaveOneGroupOut()

print(f"  Samples: {len(common)}, ASVs: {X_raw.shape[1]}, Subjects: {groups.nunique()}")

# =============================================================================
# FEATURE TRANSFORMATIONS
# =============================================================================
print("\n  Preparing feature transformations...")

# 1. TSS (relative abundance)
X_tss = X_raw.div(X_raw.sum(axis=1), axis=0).fillna(0)

# 2. CLR (centered log-ratio) — proper compositional transform
def clr_transform(df):
    """Centered log-ratio transformation with pseudocount"""
    pseudo = df + 0.5  # pseudocount to handle zeros
    log_data = np.log(pseudo)
    geometric_mean = log_data.mean(axis=1)
    return log_data.subtract(geometric_mean, axis=0)

X_clr = clr_transform(X_raw)

# 3. Genus-level TSS
genus_map = taxonomy['genus'].replace('NA', np.nan)
X_genus = X_raw.copy()
X_genus.columns = [genus_map.get(c, c) for c in X_genus.columns]
X_genus = X_genus.T.groupby(level=0).sum().T
X_genus = X_genus.loc[:, X_genus.columns.notna()]
X_genus_tss = X_genus.div(X_genus.sum(axis=1), axis=0).fillna(0)

# 4. CLR at genus level
X_genus_clr = clr_transform(X_genus)

# 5. PCA on CLR (reduce to 20 components)
pca = PCA(n_components=20, random_state=42)
scaler = StandardScaler()
X_clr_scaled = scaler.fit_transform(X_clr)
X_pca = pd.DataFrame(pca.fit_transform(X_clr_scaled),
                     index=X_clr.index,
                     columns=[f'PC{i+1}' for i in range(20)])
print(f"  PCA variance explained (20 PCs): {pca.explained_variance_ratio_.sum()*100:.1f}%")

feature_sets = {
    'ASV_TSS':       X_tss,
    'ASV_CLR':       X_clr,
    'Genus_TSS':     X_genus_tss,
    'Genus_CLR':     X_genus_clr,
    'PCA20_CLR':     X_pca,
}

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

# Classification models
clf_models = {
    'RandomForest':       RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1),
    'ExtraTrees':         ExtraTreesClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1),
    'GradientBoosting':   GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    'XGBoost':            xgb.XGBClassifier(n_estimators=200, max_depth=3, use_label_encoder=False,
                                             eval_metric='mlogloss', random_state=42, verbosity=0),
    'AdaBoost':           AdaBoostClassifier(n_estimators=200, random_state=42),
    'LogisticReg_L2':     LogisticRegression(C=1.0, penalty='l2', max_iter=2000, class_weight='balanced',
                                              random_state=42, solver='lbfgs', multi_class='multinomial'),
    'LogisticReg_L1':     LogisticRegression(C=1.0, penalty='l1', max_iter=2000, class_weight='balanced',
                                              random_state=42, solver='saga', multi_class='multinomial'),
    'RidgeClassifier':    RidgeClassifier(alpha=1.0, class_weight='balanced'),
    'SVM_RBF':            SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42),
    'SVM_Linear':         SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42),
    'KNN_5':              KNeighborsClassifier(n_neighbors=5),
    'KNN_10':             KNeighborsClassifier(n_neighbors=10),
    'MLP_small':          MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42,
                                        early_stopping=True, validation_fraction=0.15),
    'MLP_large':          MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42,
                                        early_stopping=True, validation_fraction=0.15),
}

# Regression models
reg_models = {
    'RandomForest':       RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
    'ExtraTrees':         ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1),
    'GradientBoosting':   GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
    'XGBoost':            xgb.XGBRegressor(n_estimators=200, max_depth=3, random_state=42,
                                            objective='reg:squarederror', verbosity=0),
    'AdaBoost':           AdaBoostRegressor(n_estimators=200, random_state=42),
    'Ridge':              Ridge(alpha=1.0),
    'Lasso':              Lasso(alpha=0.1, max_iter=5000, random_state=42),
    'ElasticNet':         ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42),
    'BayesianRidge':      BayesianRidge(),
    'HuberRegressor':     HuberRegressor(max_iter=500),
    'SVR_RBF':            SVR(kernel='rbf', C=1.0),
    'SVR_Linear':         SVR(kernel='linear', C=1.0),
    'KNN_5':              KNeighborsRegressor(n_neighbors=5),
    'KNN_10':             KNeighborsRegressor(n_neighbors=10),
    'MLP_small':          MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42,
                                       early_stopping=True, validation_fraction=0.15),
    'MLP_large':          MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42,
                                       early_stopping=True, validation_fraction=0.15),
    'PLS_5':              PLSRegression(n_components=5),
    'PLS_10':             PLSRegression(n_components=10),
}

# Models that need scaling (linear, SVM, KNN, MLP, PLS)
needs_scaling = {'LogisticReg_L2', 'LogisticReg_L1', 'RidgeClassifier',
                 'SVM_RBF', 'SVM_Linear', 'KNN_5', 'KNN_10',
                 'MLP_small', 'MLP_large', 'Ridge', 'Lasso', 'ElasticNet',
                 'BayesianRidge', 'HuberRegressor', 'SVR_RBF', 'SVR_Linear',
                 'PLS_5', 'PLS_10'}

# =============================================================================
# RUN CLASSIFICATION BENCHMARK
# =============================================================================
print("\n" + "=" * 70)
print("CLASSIFICATION BENCHMARK")
print("=" * 70)

# Encode labels for XGBoost
le = LabelEncoder()
y_phase_encoded = le.fit_transform(y_phase)

clf_results = []
total_combos = len(clf_models) * len(feature_sets)
combo_i = 0

for feat_name, X_feat in feature_sets.items():
    for model_name, model in clf_models.items():
        combo_i += 1
        try:
            # Build pipeline with scaling if needed
            if model_name in needs_scaling:
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            else:
                pipe = Pipeline([('model', model)])

            # XGBoost needs numeric labels
            if 'XGBoost' in model_name:
                y_target = y_phase_encoded
            else:
                y_target = y_phase

            y_pred = cross_val_predict(pipe, X_feat, y_target, groups=groups, cv=logo)

            if 'XGBoost' in model_name:
                y_pred_labels = le.inverse_transform(y_pred)
                acc = accuracy_score(y_phase, y_pred_labels)
            else:
                acc = accuracy_score(y_target, y_pred)

            clf_results.append({
                'Model': model_name,
                'Features': feat_name,
                'LOSO_Accuracy': acc,
                'N_Features': X_feat.shape[1]
            })

            if combo_i % 10 == 0 or acc > 0.5:
                print(f"  [{combo_i}/{total_combos}] {model_name:20s} + {feat_name:12s} → Acc = {acc:.3f}")
        except Exception as e:
            print(f"  [{combo_i}/{total_combos}] {model_name:20s} + {feat_name:12s} → FAILED: {str(e)[:60]}")
            clf_results.append({
                'Model': model_name, 'Features': feat_name,
                'LOSO_Accuracy': np.nan, 'N_Features': X_feat.shape[1]
            })

clf_df = pd.DataFrame(clf_results).sort_values('LOSO_Accuracy', ascending=False)
clf_df.to_csv(os.path.join(tab_dir, "multimodel_classification_benchmark.csv"), index=False)

print(f"\n  Top 10 Classification Combinations:")
print(clf_df.head(10).to_string(index=False))

# =============================================================================
# RUN REGRESSION BENCHMARK
# =============================================================================
print("\n" + "=" * 70)
print("REGRESSION BENCHMARK")
print("=" * 70)

reg_results = []
total_combos = len(reg_models) * len(feature_sets)
combo_i = 0

for feat_name, X_feat in feature_sets.items():
    for model_name, model in reg_models.items():
        combo_i += 1
        try:
            if model_name in needs_scaling:
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            else:
                pipe = Pipeline([('model', model)])

            # PLS returns 2D array
            y_pred = cross_val_predict(pipe, X_feat, y_day, groups=groups, cv=logo)
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()

            r2 = r2_score(y_day, y_pred)
            mae = mean_absolute_error(y_day, y_pred)

            reg_results.append({
                'Model': model_name,
                'Features': feat_name,
                'LOSO_R2': r2,
                'LOSO_MAE': mae,
                'N_Features': X_feat.shape[1]
            })

            if combo_i % 10 == 0 or r2 > 0.05:
                print(f"  [{combo_i}/{total_combos}] {model_name:20s} + {feat_name:12s} → R²={r2:.3f}, MAE={mae:.0f}d")
        except Exception as e:
            print(f"  [{combo_i}/{total_combos}] {model_name:20s} + {feat_name:12s} → FAILED: {str(e)[:60]}")
            reg_results.append({
                'Model': model_name, 'Features': feat_name,
                'LOSO_R2': np.nan, 'LOSO_MAE': np.nan, 'N_Features': X_feat.shape[1]
            })

reg_df = pd.DataFrame(reg_results).sort_values('LOSO_R2', ascending=False)
reg_df.to_csv(os.path.join(tab_dir, "multimodel_regression_benchmark.csv"), index=False)

print(f"\n  Top 10 Regression Combinations:")
print(reg_df.head(10).to_string(index=False))

# =============================================================================
# VISUALIZATION — Heatmaps
# =============================================================================
print("\n  Generating heatmap visualizations...")

# --- Classification Heatmap ---
clf_pivot = clf_df.pivot_table(index='Model', columns='Features', values='LOSO_Accuracy')
# Sort by mean accuracy
clf_pivot = clf_pivot.loc[clf_pivot.mean(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(clf_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.35,
            vmin=0.15, vmax=0.55, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'LOSO Accuracy'})
ax.set_title('Phase Classification — Multi-Model × Multi-Feature LOSO Benchmark',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Feature Transformation', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "multimodel_clf_heatmap.pdf"))
plt.savefig(os.path.join(fig_dir, "multimodel_clf_heatmap.png"), dpi=150)
plt.close()

# --- Regression Heatmap ---
reg_pivot = reg_df.pivot_table(index='Model', columns='Features', values='LOSO_R2')
reg_pivot = reg_pivot.loc[reg_pivot.mean(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(reg_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.0,
            vmin=-0.5, vmax=0.3, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'LOSO R²'})
ax.set_title('Temporal Regression — Multi-Model × Multi-Feature LOSO Benchmark',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Feature Transformation', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "multimodel_reg_heatmap.pdf"))
plt.savefig(os.path.join(fig_dir, "multimodel_reg_heatmap.png"), dpi=150)
plt.close()

# --- Combined bar chart: best model per category ---
clf_valid = clf_df.dropna(subset=['LOSO_Accuracy'])
reg_valid = reg_df.dropna(subset=['LOSO_R2'])
# Filter out extreme negative R2 (diverged models)
reg_valid = reg_valid[reg_valid['LOSO_R2'] > -5]
best_per_model_clf = clf_valid.loc[clf_valid.groupby('Model')['LOSO_Accuracy'].idxmax()].sort_values('LOSO_Accuracy', ascending=True)
best_per_model_reg = reg_valid.loc[reg_valid.groupby('Model')['LOSO_R2'].idxmax()].sort_values('LOSO_R2', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Classification
colors_clf = ['#2ca02c' if v > 0.45 else '#1f77b4' if v > 0.35 else '#d62728'
              for v in best_per_model_clf['LOSO_Accuracy']]
axes[0].barh(range(len(best_per_model_clf)), best_per_model_clf['LOSO_Accuracy'],
             color=colors_clf, edgecolor='white', height=0.7)
axes[0].set_yticks(range(len(best_per_model_clf)))
axes[0].set_yticklabels([f"{row['Model']}\n({row['Features']})" for _, row in best_per_model_clf.iterrows()],
                        fontsize=8)
axes[0].axvline(0.355, color='gray', linestyle='--', alpha=0.7, label='Chance (~35.5%)')
axes[0].set_xlabel('LOSO Accuracy', fontsize=11)
axes[0].set_title('Best Feature Set per Model\n(Classification)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
for i, v in enumerate(best_per_model_clf['LOSO_Accuracy']):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

# Regression
colors_reg = ['#2ca02c' if v > 0.1 else '#1f77b4' if v > 0 else '#d62728'
              for v in best_per_model_reg['LOSO_R2']]
axes[1].barh(range(len(best_per_model_reg)), best_per_model_reg['LOSO_R2'],
             color=colors_reg, edgecolor='white', height=0.7)
axes[1].set_yticks(range(len(best_per_model_reg)))
axes[1].set_yticklabels([f"{row['Model']}\n({row['Features']})" for _, row in best_per_model_reg.iterrows()],
                        fontsize=8)
axes[1].axvline(0.0, color='gray', linestyle='--', alpha=0.7, label='R²=0 (chance)')
axes[1].set_xlabel('LOSO R²', fontsize=11)
axes[1].set_title('Best Feature Set per Model\n(Temporal Regression)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
for i, v in enumerate(best_per_model_reg['LOSO_R2']):
    axes[1].text(max(v + 0.01, 0.01), i, f'{v:.3f}', va='center', fontsize=8)

plt.suptitle('Multi-Model LOSO Benchmark — All Models × 5 Feature Transforms',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "multimodel_best_per_model.pdf"))
plt.savefig(os.path.join(fig_dir, "multimodel_best_per_model.png"), dpi=150)
plt.close()

# =============================================================================
# BEST MODEL DETAILED ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("DETAILED ANALYSIS OF BEST MODELS")
print("=" * 70)

# Best classifier
best_clf_row = clf_valid.iloc[0]
print(f"\n  Best Classifier: {best_clf_row['Model']} + {best_clf_row['Features']}")
print(f"  LOSO Accuracy: {best_clf_row['LOSO_Accuracy']:.4f}")

# Re-run best classifier to get per-fold + confusion matrix
best_feat_clf = feature_sets[best_clf_row['Features']]
best_model_name = best_clf_row['Model']
best_model_clf = clf_models[best_model_name]

if best_model_name in needs_scaling:
    best_pipe_clf = Pipeline([('scaler', StandardScaler()), ('model', best_model_clf)])
else:
    best_pipe_clf = Pipeline([('model', best_model_clf)])

if 'XGBoost' in best_model_name:
    y_pred_best = cross_val_predict(best_pipe_clf, best_feat_clf, y_phase_encoded, groups=groups, cv=logo)
    y_pred_best_labels = le.inverse_transform(y_pred_best)
else:
    y_pred_best_labels = cross_val_predict(best_pipe_clf, best_feat_clf, y_phase, groups=groups, cv=logo)

print("\n  Confusion Matrix (best classifier):")
phase_order = ['Pre-Mission', 'Early', 'Mid', 'Late', 'Post-Mission']
print(classification_report(y_phase, y_pred_best_labels))

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_phase, y_pred_best_labels, labels=phase_order)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=phase_order)
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title(f'Best Classifier: {best_clf_row["Model"]} + {best_clf_row["Features"]}\n'
             f'LOSO Accuracy = {best_clf_row["LOSO_Accuracy"]:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "multimodel_best_clf_cm.pdf"))
plt.savefig(os.path.join(fig_dir, "multimodel_best_clf_cm.png"), dpi=150)
plt.close()

# Best regressor — predicted vs actual
best_reg_row = reg_valid.iloc[0]
print(f"\n  Best Regressor: {best_reg_row['Model']} + {best_reg_row['Features']}")
print(f"  LOSO R²: {best_reg_row['LOSO_R2']:.4f}, MAE: {best_reg_row['LOSO_MAE']:.1f} days")

best_feat_reg = feature_sets[best_reg_row['Features']]
best_model_reg = reg_models[best_reg_row['Model']]

if best_reg_row['Model'] in needs_scaling:
    best_pipe_reg = Pipeline([('scaler', StandardScaler()), ('model', best_model_reg)])
else:
    best_pipe_reg = Pipeline([('model', best_model_reg)])

y_pred_best_reg = cross_val_predict(best_pipe_reg, best_feat_reg, y_day, groups=groups, cv=logo)
if y_pred_best_reg.ndim > 1:
    y_pred_best_reg = y_pred_best_reg.ravel()

fig, ax = plt.subplots(figsize=(8, 8))
crew_colors = dict(zip(sorted(groups.unique()), sns.color_palette('Dark2', 6)))
for crew in sorted(groups.unique()):
    mask = groups == crew
    ax.scatter(y_day[mask], y_pred_best_reg[mask], c=[crew_colors[crew]]*mask.sum(),
               s=60, alpha=0.7, edgecolors='white', linewidth=0.5, label=crew, zorder=3)
mn, mx = min(y_day.min(), min(y_pred_best_reg)), max(y_day.max(), max(y_pred_best_reg))
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, alpha=0.7, label='Perfect')
ax.set_xlabel('Actual Day', fontsize=12)
ax.set_ylabel('Predicted Day', fontsize=12)
ax.set_title(f'Best Regressor: {best_reg_row["Model"]} + {best_reg_row["Features"]}\n'
             f'LOSO R² = {best_reg_row["LOSO_R2"]:.3f}, MAE = {best_reg_row["LOSO_MAE"]:.0f}d',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "multimodel_best_reg_scatter.pdf"))
plt.savefig(os.path.join(fig_dir, "multimodel_best_reg_scatter.png"), dpi=150)
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n  Models tested:           {len(clf_models)} classifiers × {len(reg_models)} regressors")
print(f"  Feature transforms:      {len(feature_sets)}")
print(f"  Total combinations:      {len(clf_df) + len(reg_df)}")
print(f"\n  CLASSIFICATION (best):   {best_clf_row['Model']} + {best_clf_row['Features']} → {best_clf_row['LOSO_Accuracy']:.3f}")
print(f"  REGRESSION (best):       {best_reg_row['Model']} + {best_reg_row['Features']} → R²={best_reg_row['LOSO_R2']:.3f}")

n_above_chance_clf = (clf_df['LOSO_Accuracy'] > 0.355).sum()
n_positive_r2 = (reg_df['LOSO_R2'] > 0).sum()
print(f"\n  Classifiers above chance: {n_above_chance_clf}/{len(clf_df)} ({n_above_chance_clf/len(clf_df)*100:.0f}%)")
print(f"  Regressors with R²>0:    {n_positive_r2}/{len(reg_df)} ({n_positive_r2/len(reg_df)*100:.0f}%)")

print(f"\n  Results saved to:")
print(f"    {os.path.join(tab_dir, 'multimodel_classification_benchmark.csv')}")
print(f"    {os.path.join(tab_dir, 'multimodel_regression_benchmark.csv')}")
print(f"    {fig_dir}/multimodel_*.png")
print("\n" + "=" * 70)
