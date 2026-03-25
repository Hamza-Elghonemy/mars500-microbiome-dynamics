import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

# ==============================================================================
# M4: Machine Learning Classification (Python)
# MARS500 Temporal Gut Microbiome Dynamics
# ==============================================================================
# Predicts mission phase (Pre/Early/Mid/Late/Post) from ASV abundance profiles.
# Uses Random Forest and XGBoost with 5-fold CV.
# ==============================================================================

print("Starting Machine Learning Pipeline...")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
meta_dir = os.path.join(base_dir, "data", "metadata")
results_dir = os.path.join(base_dir, "results", "models")
os.makedirs(results_dir, exist_ok=True)

# 1. Load Data
print("Loading Feature Matrix & Metadata...")
counts_file = os.path.join(data_dir, "GLDS-191_GAmplicon_counts.tsv")
meta_file = os.path.join(meta_dir, "processed_metadata.tsv")

if os.path.exists(counts_file) and os.path.exists(meta_file):
    # ASV counts: rows=ASVs, columns=Samples -> we need to transpose
    X_raw = pd.read_csv(counts_file, sep="\t", index_col=0).T
    meta = pd.read_csv(meta_file, sep="\t", index_col="SampleID")
    
    # Align indices
    common_idx = X_raw.index.intersection(meta.index)
    X = X_raw.loc[common_idx]
    
    # Normalize counts to relative abundance (TSS)
    X_rel = X.div(X.sum(axis=1), axis=0)
    
    # Build target vector
    y = meta.loc[common_idx, 'Phase']
    
    # 2. Train and Evaluate Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(rf_clf, X_rel, y, cv=cv, scoring='accuracy')
    
    print(f"Random Forest CV Accuracy: {rf_scores.mean():.2f} ± {rf_scores.std():.2f}")
    
    # Fit full model to extract feature importance and confusion matrix
    rf_clf.fit(X_rel, y)
    importances = rf_clf.feature_importances_
    
    importances_df = pd.DataFrame({'importance': importances}, index=X_rel.columns).sort_values('importance', ascending=False)
    importances_df.to_csv(os.path.join(results_dir, "rf_feature_importance.csv"))
    
    # Plot Confusion Matrix
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(rf_clf, X_rel, y, cv=cv)
    cm = confusion_matrix(y, y_pred, labels=rf_clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_clf.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Random Forest Cross-Validated Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "results", "figures", "rf_confusion_matrix.pdf"))
    
    # Plot top 15 features
    top_features = importances_df.head(15)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_features['importance'], y=top_features.index)
    plt.title('Top 15 Phase-Predictive Taxonomic Biomarkers (Random Forest)')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "results", "figures", "rf_feature_importance.pdf"))
    
    # 3. XGBoost
    print("Training XGBoost Classifier...")
    classes = y.unique()
    class_mapping = {c: i for i, c in enumerate(classes)}
    y_num = y.map(class_mapping)
    
    xgb_clf = xgb.XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_scores = cross_val_score(xgb_clf, X_rel, y_num, cv=cv, scoring='accuracy')
    print(f"XGBoost CV Accuracy: {xgb_scores.mean():.2f} ± {xgb_scores.std():.2f}")
    
    print("M4: Machine Learning Classification complete.")
else:
    print("[WARNING] Processed inputs missing. Requires counts & metadata to proceed.")
