import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# ==============================================================================
# M8: Temporal Machine Learning Regression (Python)
# MARS500 Temporal Gut Microbiome Dynamics (Phase IV)
# ==============================================================================
# Predicts exact `Timepoint_Day` (continuous) from ASV abundance profiles.
# Resolves categorical ML limitations while avoiding LSTM overfitting.
# ==============================================================================

print("Starting Temporal Machine Learning Regressors...")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
meta_dir = os.path.join(base_dir, "data", "metadata")
results_dir = os.path.join(base_dir, "results", "models")
fig_dir = os.path.join(base_dir, "results", "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# 1. Load Data
counts_file = os.path.join(data_dir, "GLDS-191_GAmplicon_counts.tsv")
meta_file = os.path.join(meta_dir, "processed_metadata.tsv")
taxa_file = os.path.join(data_dir, "GLDS-191_GAmplicon_taxonomy.tsv")

if not (os.path.exists(counts_file) and os.path.exists(meta_file)):
    print("[WARNING] Required data files are missing.")
    exit(1)

X_raw = pd.read_csv(counts_file, sep="\t", index_col=0).T
meta = pd.read_csv(meta_file, sep="\t", index_col="SampleID")

# Load taxonomy to map ASV -> Genus for prettier plots
taxa = pd.read_csv(taxa_file, sep="\t", index_col=0)
taxa_dict = {}
for idx, row in taxa.iterrows():
    # Extract Genus (assume format like d__Bacteria;...;g__Bacteroides)
    try:
        t_str = str(row['taxonomy'])
        genus = [x for x in t_str.split(';') if x.strip().startswith('g__')]
        name = genus[0].split('__')[1] if genus else idx
        taxa_dict[idx] = name
    except:
        taxa_dict[idx] = idx

common_idx = X_raw.index.intersection(meta.index)
X = X_raw.loc[common_idx]
X_rel = X.div(X.sum(axis=1), axis=0) # Total Sum Scaling

# Target is now continuous Timepoint_Day!
y = meta.loc[common_idx, 'Timepoint_Day'].astype(float)

# 2. Random Forest Regressor
print("Training Random Forest Temporal Vector...")
rf_reg = RandomForestRegressor(n_estimators=500, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate via MAE & R2
rf_mae = -cross_val_score(rf_reg, X_rel, y, cv=cv, scoring='neg_mean_absolute_error')
rf_r2 = cross_val_score(rf_reg, X_rel, y, cv=cv, scoring='r2')

print(f"Random Forest CV MAE: {rf_mae.mean():.1f} days ± {rf_mae.std():.1f}")
print(f"Random Forest CV R²: {rf_r2.mean():.2f} ± {rf_r2.std():.2f}")

# Extract out-of-fold predictions
y_pred_rf = cross_val_predict(rf_reg, X_rel, y, cv=cv)

# 3. XGBoost Regressor
print("Training XGBoost Temporal Vector...")
xgb_reg = xgb.XGBRegressor(n_estimators=500, random_state=42, objective='reg:squarederror')
xgb_mae = -cross_val_score(xgb_reg, X_rel, y, cv=cv, scoring='neg_mean_absolute_error')
xgb_r2 = cross_val_score(xgb_reg, X_rel, y, cv=cv, scoring='r2')

print(f"XGBoost CV MAE: {xgb_mae.mean():.1f} days ± {xgb_mae.std():.1f}")
print(f"XGBoost CV R²: {xgb_r2.mean():.2f} ± {xgb_r2.std():.2f}")

# 4. Fit Full RF and Extract Feature Importance
rf_reg.fit(X_rel, y)
importances = rf_reg.feature_importances_
importances_df = pd.DataFrame({'importance': importances}, index=X_rel.columns).sort_values('importance', ascending=False)
importances_df.to_csv(os.path.join(results_dir, "temporal_rf_feature_importance.csv"))

# Map top 15 ASVs to their Taxonomy Genera names
top_features = importances_df.head(15).copy()
top_features.index = [f"{taxa_dict.get(asv, asv)} ({asv})" for asv in top_features.index]

plt.figure(figsize=(10, 8))
sns.barplot(x=top_features['importance'], y=top_features.index, palette='viridis')
plt.title('Top 15 Strict-Temporal Taxonomic Biomarkers (Age-of-Isolation)')
plt.xlabel('Gini Importance (Node Purity Reduction)')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "temporal_rf_biomarkers.pdf"))

# 5. Plot Predicted vs Actual Isolation Day
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y, y_pred_rf, alpha=0.8, c='royalblue', edgecolor='w', s=100)
# Plot ideal perfect prediction line (y = x)
min_val, max_val = min(y.min(), y_pred_rf.min()), max(y.max(), y_pred_rf.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Synchronization (y=x)")

ax.set_xlabel("Actual Isolation Day", fontsize=12)
ax.set_ylabel("Microbiome-Predicted Isolation Day", fontsize=12)
ax.set_title(f"Temporal Regression: Predicting Subject Timeline\n(Random Forest CV R² = {rf_r2.mean():.2f}, MAE = ±{rf_mae.mean():.0f} Days)")
ax.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "temporal_predicted_vs_actual.pdf"))
print("Temporal plots generated securely into results/figures/")
