# ==============================================================================
# Pipeline 28: LOSO Fold-wise Binomial 95% Confidence Intervals
# Determines whether per-fold accuracy differences are within sampling variance
# ==============================================================================

import os
import pandas as pd
import numpy as np
from scipy.stats import binom
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Pipeline 28: LOSO Fold-wise Confidence Intervals")
print("=" * 60)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tab_dir  = os.path.join(base_dir, "results", "tables")

# Load existing per-fold results
fold_df = pd.read_csv(os.path.join(tab_dir, "loso_classification_per_fold.csv"))
print("Loaded per-fold data:")
print(fold_df)

def wilson_ci(k, n, confidence=0.95):
    """Wilson score confidence interval for a proportion."""
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    center = (p_hat + z**2 / (2*n)) / (1 + z**2 / n)
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return max(0, center - margin), min(1, center + margin)

# Compute number of correct predictions from accuracy x n_test
fold_df['N_correct'] = (fold_df['Accuracy'] * fold_df['N_test']).round().astype(int)

rows = []
for _, row in fold_df.iterrows():
    k = int(row['N_correct'])
    n = int(row['N_test'])
    acc = row['Accuracy']
    lo, hi = wilson_ci(k, n)
    rows.append({
        'CrewMember':    row['CrewMember'],
        'N_test':        n,
        'N_correct':     k,
        'Accuracy':      round(acc, 4),
        'Accuracy_Pct':  f"{acc:.1%}",
        'CI_Lower':      round(lo, 4),
        'CI_Upper':      round(hi, 4),
        'CI_Lower_Pct':  f"{lo:.1%}",
        'CI_Upper_Pct':  f"{hi:.1%}",
    })

ci_df = pd.DataFrame(rows)

# Chance level for 5-class problem
n_classes   = 5  # Pre-Mission, Early, Mid, Late, Post-Mission
chance_level = 1 / n_classes
mean_acc     = fold_df['Accuracy'].mean()

ci_df['Above_chance'] = ci_df['CI_Lower'] > chance_level
ci_df['Notes'] = ''
ci_df.loc[ci_df['Accuracy'] == ci_df['Accuracy'].max(), 'Notes'] = 'Highest fold'
ci_df.loc[ci_df['Accuracy'] == ci_df['Accuracy'].min(), 'Notes'] = 'Lowest fold'

print("\n=== Per-Fold Results with 95% Wilson CIs ===")
print(ci_df[['CrewMember', 'N_test', 'Accuracy_Pct', 'CI_Lower_Pct', 'CI_Upper_Pct', 'Above_chance', 'Notes']].to_string(index=False))
print(f"\nMean accuracy: {mean_acc:.1%}")
print(f"Chance level ({n_classes} classes): {chance_level:.1%}")
print(f"Folds above chance (CI_lower > {chance_level:.1%}): {ci_df['Above_chance'].sum()}/{len(ci_df)}")

# Check if S6 CI overlaps with S3 CI (are they statistically distinct?)
s6 = ci_df[ci_df['CrewMember'] == 'S6'].iloc[0]
s3 = ci_df[ci_df['CrewMember'] == 'S3'].iloc[0]
overlap = s6['CI_Upper'] >= s3['CI_Lower']
print(f"\nS3 95% CI: [{s3['CI_Lower_Pct']}, {s3['CI_Upper_Pct']}]")
print(f"S6 95% CI: [{s6['CI_Lower_Pct']}, {s6['CI_Upper_Pct']}]")
print(f"CIs overlap (signal is distributed, not dominated): {overlap}")

ci_df.to_csv(os.path.join(tab_dir, "loso_foldwise_with_ci.csv"), index=False)
print(f"\nSaved to results/tables/loso_foldwise_with_ci.csv")
