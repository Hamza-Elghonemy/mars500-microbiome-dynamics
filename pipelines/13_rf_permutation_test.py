import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ==============================================================================
# W1.4: Random Forest Permutation Validation
# MARS500 Publication Pipeline
# ==============================================================================

print("W1.4: Random Forest Permutation Importance Test...")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
meta_dir = os.path.join(base_dir, "data", "metadata")
results_dir = os.path.join(base_dir, "results", "models")
tab_dir = os.path.join(base_dir, "results", "tables")
fig_dir = os.path.join(base_dir, "results", "figures")
os.makedirs(results_dir, exist_ok=True)

# Load
X_raw = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_counts.tsv"), sep="\t", index_col=0).T
meta = pd.read_csv(os.path.join(meta_dir, "processed_metadata.tsv"), sep="\t", index_col="SampleID")

common_idx = X_raw.index.intersection(meta.index)
X = X_raw.loc[common_idx]
X_rel = X.div(X.sum(axis=1), axis=0)
y = meta.loc[common_idx, 'Phase']

# Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)

# 1. Cross-validated predictions for confusion matrix
print("  → Cross-validated confusion matrix...")
y_pred = cross_val_predict(rf, X_rel, y, cv=cv)

# Save confusion matrix as CSV
labels = ["Pre-Mission", "Early", "Mid", "Late", "Post-Mission"]
cm = confusion_matrix(y, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(os.path.join(tab_dir, "rf_confusion_matrix.csv"))

# Save classification report
report = classification_report(y, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).T
report_df.to_csv(os.path.join(tab_dir, "rf_classification_report.csv"))
print(classification_report(y, y_pred, target_names=labels))

# 2. Permutation test (1000 permutations)
print("  → Permutation test (1000 iterations, this takes a few minutes)...")
score, perm_scores, pvalue = permutation_test_score(
    rf, X_rel, y, cv=cv, scoring='accuracy',
    n_permutations=1000, random_state=42, n_jobs=-1
)

print(f"  True CV Accuracy: {score:.3f}")
print(f"  Permutation p-value: {pvalue:.4f}")
print(f"  Mean null accuracy: {perm_scores.mean():.3f} ± {perm_scores.std():.3f}")

# Save permutation results
perm_df = pd.DataFrame({
    'true_accuracy': [score],
    'mean_null_accuracy': [perm_scores.mean()],
    'std_null_accuracy': [perm_scores.std()],
    'p_value': [pvalue],
    'n_permutations': [1000]
})
perm_df.to_csv(os.path.join(tab_dir, "rf_permutation_test.csv"), index=False)

# Plot permutation null distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(perm_scores, bins=40, color='#cccccc', edgecolor='white', label='Null Distribution')
ax.axvline(score, color='#D7191C', lw=2, linestyle='--', label=f'True Accuracy = {score:.2f}')
ax.axvline(perm_scores.mean(), color='#2C7BB6', lw=1.5, linestyle=':', label=f'Null Mean = {perm_scores.mean():.2f}')
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'RF Permutation Test (p = {pvalue:.4f}, n = 1000)', fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "rf_permutation_test.pdf"))
print("W1.4: RF validation complete. Results in results/tables/")
