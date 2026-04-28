import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# WP2: DA Robustness — DESeq2 vs ANCOM-BC2 Concordance
# MARS500 Publication Pipeline
# ==============================================================================
# Loads existing DESeq2 and ANCOM-BC2 results and builds a unified concordance
# table showing which taxa claims are robust across methods.
#
# For each focal taxon, reports:
#   - Direction (up/down in Late vs Pre-Mission)
#   - Effect size (LFC)
#   - Adjusted p-value
#   - Whether signal is consistent across methods
#   - Final verdict: Robust / Weak / Discordant / Unsupported
# ==============================================================================

print("=" * 70)
print("WP2: Differential Abundance Robustness Check")
print("=" * 70)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tab_dir  = os.path.join(base_dir, "results", "tables")
fig_dir  = os.path.join(base_dir, "results", "figures")

# --- Load DESeq2 results ---
deseq_file = os.path.join(tab_dir, "DESeq2_Late_vs_PreMission.csv")
ancom_file = os.path.join(tab_dir, "ANCOMBC_Late_vs_PreMission.csv")

if not os.path.exists(deseq_file):
    print(f"  [ERROR] DESeq2 results not found: {deseq_file}")
    print(f"  Run pipeline 06 first.")
    exit(1)

deseq = pd.read_csv(deseq_file, index_col=0)
print(f"  DESeq2 results: {len(deseq)} taxa")

# --- Load ANCOM-BC2 results ---
if os.path.exists(ancom_file):
    ancom = pd.read_csv(ancom_file)
    print(f"  ANCOM-BC2 results: {len(ancom)} taxa")
    has_ancom = True
else:
    print(f"  [WARNING] ANCOM-BC2 results not found. Running concordance with DESeq2 only.")
    has_ancom = False

# --- Focal taxa from the paper narrative ---
focal_taxa = ['Faecalibacterium', 'Sutterella', 'Bilophila', 'Akkermansia',
              'Bacteroides', 'Dialister', 'Roseburia', 'Ruminococcus',
              'Lachnospira', 'Blautia', 'Coprococcus', 'Dorea',
              'Oscillospira', 'Prevotella', 'Parabacteroides']

print(f"\n  Focal taxa: {len(focal_taxa)}")

# =============================================================================
# BUILD CONCORDANCE TABLE
# =============================================================================
concordance = []

for genus in focal_taxa:
    row = {'Genus': genus}

    # --- DESeq2 ---
    deseq_match = deseq[deseq['Genus'].astype(str).str.lower() == genus.lower()]
    if len(deseq_match) > 0:
        best = deseq_match.sort_values('padj').iloc[0]
        row['DESeq2_LFC'] = best['log2FoldChange']
        row['DESeq2_padj'] = best['padj']
        row['DESeq2_Direction'] = 'Up' if best['log2FoldChange'] > 0 else 'Down'
        row['DESeq2_Significant'] = best['padj'] < 0.05 if pd.notna(best['padj']) else False
    else:
        row['DESeq2_LFC'] = np.nan
        row['DESeq2_padj'] = np.nan
        row['DESeq2_Direction'] = 'Not tested'
        row['DESeq2_Significant'] = False

    # --- ANCOM-BC2 ---
    if has_ancom:
        ancom_match = ancom[ancom['Genus'].astype(str).str.lower() == genus.lower()]
        if len(ancom_match) > 0:
            best_a = ancom_match.iloc[0]
            row['ANCOMBC_LFC'] = best_a['LFC']
            row['ANCOMBC_qval'] = best_a['q_value']
            row['ANCOMBC_Direction'] = 'Up' if best_a['LFC'] > 0 else 'Down'
            row['ANCOMBC_Significant'] = bool(best_a['Significant']) if pd.notna(best_a.get('Significant')) else best_a['q_value'] < 0.05
        else:
            row['ANCOMBC_LFC'] = np.nan
            row['ANCOMBC_qval'] = np.nan
            row['ANCOMBC_Direction'] = 'Not tested'
            row['ANCOMBC_Significant'] = False
    else:
        row['ANCOMBC_LFC'] = np.nan
        row['ANCOMBC_qval'] = np.nan
        row['ANCOMBC_Direction'] = 'N/A'
        row['ANCOMBC_Significant'] = False

    # --- Concordance Verdict ---
    if has_ancom:
        d_sig = row['DESeq2_Significant']
        a_sig = row['ANCOMBC_Significant']
        d_dir = row['DESeq2_Direction']
        a_dir = row['ANCOMBC_Direction']
        d_tested = d_dir != 'Not tested'
        a_tested = a_dir != 'Not tested'

        if d_tested and a_tested:
            # Both methods tested this taxon
            if d_sig and a_sig and d_dir == a_dir:
                row['Verdict'] = '✅ Robust'
                row['Confidence'] = 'High'
            elif d_sig and a_sig and d_dir != a_dir:
                row['Verdict'] = '⚠️ Discordant direction'
                row['Confidence'] = 'Low'
            elif (d_sig or a_sig) and d_dir == a_dir:
                row['Verdict'] = '🟡 Weak (one method sig, same direction)'
                row['Confidence'] = 'Moderate'
            elif (d_sig or a_sig) and d_dir != a_dir:
                row['Verdict'] = '⚠️ Discordant direction'
                row['Confidence'] = 'Low'
            elif d_dir == a_dir:
                row['Verdict'] = '🔵 Trend only (same direction, neither sig)'
                row['Confidence'] = 'Low'
            else:
                row['Verdict'] = '❌ Not significant'
                row['Confidence'] = 'None'
        elif d_tested and not a_tested:
            # Only DESeq2 tested (taxon filtered out in ANCOM-BC2)
            if d_sig:
                row['Verdict'] = '🟡 DESeq2 only (ANCOM-BC2 filtered)'
                row['Confidence'] = 'Moderate'
            else:
                row['Verdict'] = '❌ Not significant (DESeq2 only)'
                row['Confidence'] = 'None'
        elif not d_tested and a_tested:
            if a_sig:
                row['Verdict'] = '🟡 ANCOM-BC2 only (DESeq2 filtered)'
                row['Confidence'] = 'Moderate'
            else:
                row['Verdict'] = '❌ Not significant (ANCOM-BC2 only)'
                row['Confidence'] = 'None'
        else:
            row['Verdict'] = '❌ Not tested by either method'
            row['Confidence'] = 'None'
    else:
        if row['DESeq2_Significant']:
            row['Verdict'] = '🟡 DESeq2 only (needs ANCOM-BC2 confirmation)'
            row['Confidence'] = 'Moderate'
        else:
            row['Verdict'] = '❌ Not significant'
            row['Confidence'] = 'None'

    concordance.append(row)

conc_df = pd.DataFrame(concordance)
conc_df.to_csv(os.path.join(tab_dir, "DA_concordance_table.csv"), index=False)

print("\n  CONCORDANCE TABLE (Late vs Pre-Mission):")
print("  " + "-" * 90)
for _, r in conc_df.iterrows():
    deseq_str = f"LFC={r['DESeq2_LFC']:+.2f}, padj={r['DESeq2_padj']:.3f}" if pd.notna(r['DESeq2_LFC']) else "Not tested"
    if has_ancom:
        ancom_str = f"LFC={r['ANCOMBC_LFC']:+.2f}, q={r['ANCOMBC_qval']:.3f}" if pd.notna(r['ANCOMBC_LFC']) else "Not tested"
    else:
        ancom_str = "N/A"
    print(f"  {r['Genus']:20s} | DESeq2: {deseq_str:35s} | ANCOM: {ancom_str:30s} | {r['Verdict']}")

# =============================================================================
# CLAIM STATUS SUMMARY
# =============================================================================
print("\n\n  CLAIM STATUS SUMMARY:")
print("  " + "=" * 60)
for verdict in ['✅ Robust', '🟡 Weak (one method only)', '⚠️ Discordant direction',
                '🔵 Trend only (same direction)', '❌ Not significant', '❓ Inconclusive']:
    subset = conc_df[conc_df['Verdict'] == verdict]
    if len(subset) > 0:
        taxa_list = ", ".join(subset['Genus'].tolist())
        print(f"  {verdict}: {taxa_list}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n  Generating concordance figure...")

if has_ancom:
    # Scatter: DESeq2 LFC vs ANCOM-BC2 LFC
    plot_df = conc_df.dropna(subset=['DESeq2_LFC', 'ANCOMBC_LFC']).copy()

    if len(plot_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Panel A: LFC comparison
        ax = axes[0]
        colors = []
        for _, r in plot_df.iterrows():
            if r['DESeq2_Significant'] and r['ANCOMBC_Significant']:
                colors.append('#2ca02c')
            elif r['DESeq2_Significant'] or r['ANCOMBC_Significant']:
                colors.append('#ff7f0e')
            else:
                colors.append('#7f7f7f')

        ax.scatter(plot_df['DESeq2_LFC'], plot_df['ANCOMBC_LFC'], c=colors, s=100,
                   edgecolors='white', linewidth=0.5, zorder=3)

        # Label focal taxa
        for _, r in plot_df.iterrows():
            if r['DESeq2_Significant'] or r['ANCOMBC_Significant']:
                ax.annotate(r['Genus'], (r['DESeq2_LFC'], r['ANCOMBC_LFC']),
                           fontsize=8, ha='left', va='bottom',
                           xytext=(5, 3), textcoords='offset points')

        # Add diagonal line and zero lines
        lim = max(abs(plot_df['DESeq2_LFC']).max(), abs(plot_df['ANCOMBC_LFC']).max()) + 1
        ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='1:1 line')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

        # Correlation
        corr, p = stats.pearsonr(plot_df['DESeq2_LFC'], plot_df['ANCOMBC_LFC'])
        ax.set_xlabel('DESeq2 log₂FC', fontsize=12)
        ax.set_ylabel('ANCOM-BC2 LFC', fontsize=12)
        ax.set_title(f'Effect Size Concordance\nr = {corr:.3f}, p = {p:.2e}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel B: Verdict summary
        ax = axes[1]
        verdict_counts = conc_df['Verdict'].value_counts()
        colors_bar = {'✅ Robust': '#2ca02c', '🟡 Weak (one method only)': '#ff7f0e',
                      '⚠️ Discordant direction': '#d62728', '🔵 Trend only (same direction)': '#1f77b4',
                      '❌ Not significant': '#7f7f7f', '❓ Inconclusive': '#bcbd22'}
        bar_colors = [colors_bar.get(v, '#7f7f7f') for v in verdict_counts.index]
        ax.barh(range(len(verdict_counts)), verdict_counts.values, color=bar_colors,
                edgecolor='white', height=0.6)
        ax.set_yticks(range(len(verdict_counts)))
        ax.set_yticklabels(verdict_counts.index, fontsize=10)
        ax.set_xlabel('Number of Taxa', fontsize=11)
        ax.set_title('DA Concordance Verdicts\n(DESeq2 vs ANCOM-BC2)', fontsize=12, fontweight='bold')
        for i, v in enumerate(verdict_counts.values):
            ax.text(v + 0.1, i, str(v), va='center', fontsize=11, fontweight='bold')

        plt.suptitle('WP2: Differential Abundance Robustness Check\nLate Confinement vs Pre-Mission',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "WP2_DA_concordance.pdf"))
        plt.savefig(os.path.join(fig_dir, "WP2_DA_concordance.png"), dpi=150)
        plt.close()
        print("  Saved concordance figure.")
    else:
        print("  [WARNING] No overlapping taxa between methods for scatter plot.")
else:
    # DESeq2-only figure: significance barplot
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = conc_df.dropna(subset=['DESeq2_LFC']).sort_values('DESeq2_LFC')
    colors = ['#2ca02c' if r['DESeq2_Significant'] else '#7f7f7f' for _, r in plot_df.iterrows()]
    ax.barh(range(len(plot_df)), plot_df['DESeq2_LFC'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Genus'], fontsize=10)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('log₂ Fold Change (Late vs Pre-Mission)', fontsize=12)
    ax.set_title('DESeq2 Differential Abundance — Focal Taxa\n(Green = padj < 0.05)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "WP2_DA_deseq2_focal_taxa.pdf"))
    plt.savefig(os.path.join(fig_dir, "WP2_DA_deseq2_focal_taxa.png"), dpi=150)
    plt.close()



# =============================================================================
# FINAL RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 70)
print("WP2 RECOMMENDATIONS FOR MANUSCRIPT")
print("=" * 70)

robust = conc_df[conc_df['Confidence'] == 'High']['Genus'].tolist()
weak = conc_df[conc_df['Confidence'] == 'Moderate']['Genus'].tolist()
unsupported = conc_df[conc_df['Confidence'].isin(['None', 'Low'])]['Genus'].tolist()

print(f"\n  SAFE TO CLAIM (robust across methods): {', '.join(robust) if robust else 'None'}")
print(f"  MENTION CAUTIOUSLY (one method only): {', '.join(weak) if weak else 'None'}")
print(f"  DO NOT CLAIM (unsupported/discordant): {', '.join(unsupported) if unsupported else 'None'}")
print("=" * 70)
