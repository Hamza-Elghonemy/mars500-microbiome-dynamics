# ==============================================================================
# Pipeline 29: Extinct-Node Taxonomic Analysis + Functional Redundancy Factor
# Addresses reviewer requests for:
# (1) taxonomic identity of nodes lost Early -> Late (Discussion 4.4)
# (2) quantified functional redundancy ratio (Results 3.5)
# ==============================================================================

import os
import gzip
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import statistics

print("=" * 60)
print("Pipeline 29: Extinct Nodes + Functional Redundancy")
print("=" * 60)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
meta_dir = os.path.join(base_dir, "data", "metadata")
tab_dir  = os.path.join(base_dir, "results", "tables")
fig_dir  = os.path.join(base_dir, "results", "figures")
nsti_dir = os.path.join(base_dir, "results", "picrust2", "EC_metagenome_out")

# -- Load data --
metadata = pd.read_csv(os.path.join(meta_dir, "processed_metadata.tsv"), sep="\t")
sample_phase = dict(zip(metadata['SampleID'], metadata['Phase']))

taxonomy = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_taxonomy.tsv"), sep="\t")
asv_tax = taxonomy.set_index('ASV_ID').to_dict('index')

centrality = pd.read_csv(os.path.join(tab_dir, "centrality_all_phases.csv"))

# -- PART A: Extinct-node analysis --
print("\n=== PART A: Extinct-Node Taxonomic Analysis ===")

phase_taxa = {}
for phase in ['Early', 'Mid', 'Late', 'Post-Mission']:
    phase_taxa[phase] = set(centrality[centrality['Phase'] == phase]['Taxon'])

lost_early_late  = phase_taxa['Early'] - phase_taxa['Late']
gained_early_late = phase_taxa['Late'] - phase_taxa['Early']
retained          = phase_taxa['Early'] & phase_taxa['Late']

print(f"Early total:     {len(phase_taxa['Early'])}")
print(f"Late total:      {len(phase_taxa['Late'])}")
print(f"Lost E->L:       {len(lost_early_late)}")
print(f"Gained E->L:     {len(gained_early_late)}")
print(f"Retained:        {len(retained)}")

# Butyrate-producing orders and primary degraders
butyrate_orders = {'Lachnospirales', 'Oscillospirales'}
degrader_orders = {'Bacteroidales'}

def order_profile(taxa_set, label):
    cnt = Counter(asv_tax.get(asv, {}).get('order', 'Unknown') for asv in taxa_set)
    total = sum(cnt.values())
    rows = [{'Label': label, 'Order': k, 'Count': v,
             'Pct': round(100 * v / total, 1)} for k, v in cnt.most_common(12)]
    return pd.DataFrame(rows)

early_profile = order_profile(phase_taxa['Early'], 'Early_baseline')
lost_profile  = order_profile(lost_early_late,     'Lost_Early_to_Late')
gained_profile = order_profile(gained_early_late,  'Gained_Early_to_Late')
extinct_profile = pd.concat([early_profile, lost_profile, gained_profile])
extinct_profile.to_csv(os.path.join(tab_dir, "extinct_node_order_profile.csv"), index=False)

# Butyrate group counts
early_but = sum(1 for a in phase_taxa['Early']
                if asv_tax.get(a, {}).get('order', '') in butyrate_orders)
lost_but  = sum(1 for a in lost_early_late
                if asv_tax.get(a, {}).get('order', '') in butyrate_orders)
gained_but = sum(1 for a in gained_early_late
                 if asv_tax.get(a, {}).get('order', '') in butyrate_orders)
early_deg = sum(1 for a in phase_taxa['Early']
                if asv_tax.get(a, {}).get('order', '') in degrader_orders)
lost_deg  = sum(1 for a in lost_early_late
                if asv_tax.get(a, {}).get('order', '') in degrader_orders)

summary_rows = [
    {'Metric': 'Total Early nodes',                      'Value': len(phase_taxa['Early'])},
    {'Metric': 'Total Late nodes',                       'Value': len(phase_taxa['Late'])},
    {'Metric': 'Nodes lost Early->Late',                 'Value': len(lost_early_late)},
    {'Metric': 'Nodes gained Early->Late',               'Value': len(gained_early_late)},
    {'Metric': 'Retained nodes',                         'Value': len(retained)},
    {'Metric': 'Early butyrate-order ASVs (Lachnospirales+Oscillospirales)',
                                                         'Value': early_but},
    {'Metric': 'Lost butyrate-order ASVs',               'Value': lost_but},
    {'Metric': 'Pct of Early butyrate ASVs lost (%)',
     'Value': round(100*lost_but/max(early_but,1), 1)},
    {'Metric': 'Pct of lost nodes from butyrate orders',
     'Value': round(100*lost_but/len(lost_early_late), 1) if lost_early_late else 0},
    {'Metric': 'Gained butyrate-order ASVs',             'Value': gained_but},
    {'Metric': 'Early Bacteroidales ASVs',               'Value': early_deg},
    {'Metric': 'Lost Bacteroidales ASVs',                'Value': lost_deg},
    {'Metric': 'Bacteroidales turnover (%)',
     'Value': round(100*lost_deg/max(early_deg,1), 1)},
]
pd.DataFrame(summary_rows).to_csv(
    os.path.join(tab_dir, "extinct_node_summary.csv"), index=False)

print(f"\nButyrate-order (Lachnospirales + Oscillospirales):")
print(f"  Early: {early_but}")
print(f"  Lost:  {lost_but}")
if early_but > 0:
    print(f"  {lost_but}/{early_but} = {100*lost_but/early_but:.1f}% lost")
print(f"Bacteroidales turnover: {lost_deg}/{early_deg}")

# -- PART B: Functional Redundancy Factor --
print("\n=== PART B: Functional Redundancy Factor ===")

nsti_gz = os.path.join(nsti_dir, "pred_metagenome_unstrat.tsv.gz")
if not os.path.exists(nsti_gz):
    print(f"WARNING: {nsti_gz} not found. Skipping functional redundancy.")
else:
    with gzip.open(nsti_gz, 'rt') as f:
        pathway_df = pd.read_csv(f, sep='\t', index_col=0)

    def shannon_entropy(counts):
        counts = counts[counts > 0]
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return -float((p * np.log(p)).sum())

    phase_func_shannon = defaultdict(list)
    for sample_id in pathway_df.columns:
        phase = sample_phase.get(sample_id, 'Unknown')
        if phase == 'Unknown':
            continue
        sh = shannon_entropy(pathway_df[sample_id].values)
        phase_func_shannon[phase].append(sh)

    # Taxonomic Shannon from phase_level_integrated_metrics
    phase_metrics = pd.read_csv(
        os.path.join(tab_dir, "phase_level_integrated_metrics.csv"))
    tax_shannon = dict(zip(phase_metrics['Phase'], phase_metrics['Shannon_mean']))

    phases_ordered = ['Pre-Mission', 'Early', 'Mid', 'Late', 'Post-Mission']
    redundancy_rows = []
    for phase in phases_ordered:
        func_vals = phase_func_shannon.get(phase, [])
        tax_sh    = tax_shannon.get(phase, None)
        if func_vals and tax_sh:
            func_sh = statistics.mean(func_vals)
            func_sd = statistics.stdev(func_vals) if len(func_vals) > 1 else 0.0
            redundancy_rows.append({
                'Phase':              phase,
                'N_samples':          len(func_vals),
                'Taxonomic_Shannon':  round(tax_sh, 3),
                'Functional_Shannon_mean': round(func_sh, 3),
                'Functional_Shannon_SD':   round(func_sd, 3),
                'Redundancy_Ratio':        round(func_sh / tax_sh, 3),
            })

    redund_df = pd.DataFrame(redundancy_rows)
    tax_range  = redund_df['Taxonomic_Shannon'].max() - redund_df['Taxonomic_Shannon'].min()
    func_range = (redund_df['Functional_Shannon_mean'].max()
                  - redund_df['Functional_Shannon_mean'].min())
    resilience_ratio = func_range / tax_range if tax_range > 0 else None

    redund_df['Tax_Shannon_range']  = round(tax_range, 3)
    redund_df['Func_Shannon_range'] = round(func_range, 3)
    if resilience_ratio is not None:
        redund_df['Resilience_ratio'] = round(resilience_ratio, 3)
    else:
        redund_df['Resilience_ratio'] = None

    redund_df.to_csv(
        os.path.join(tab_dir, "functional_redundancy_factor.csv"), index=False)

    print(redund_df[['Phase', 'Taxonomic_Shannon',
                     'Functional_Shannon_mean', 'Redundancy_Ratio']].to_string(index=False))
    print(f"\nTaxonomic Shannon range:   {tax_range:.3f}")
    print(f"Functional Shannon range:  {func_range:.3f}")
    if resilience_ratio is not None:
        print(f"Functional resilience ratio: {resilience_ratio:.3f}")
        print(f"=> Functional diversity varies "
              f"{100*resilience_ratio:.1f}% as much as taxonomic diversity")

    # -- Figure: Side-by-side diversity trajectories --
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(redund_df['Phase'], redund_df['Taxonomic_Shannon'],
                 'o-', color='steelblue', lw=2)
    axes[0].set_title("Taxonomic Shannon Entropy by Phase")
    axes[0].set_ylabel("Shannon Entropy (nat log)")
    axes[0].set_xlabel("Mission Phase")
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].plot(redund_df['Phase'], redund_df['Functional_Shannon_mean'],
                 's-', color='darkorange', lw=2)
    axes[1].set_title("Functional Shannon Entropy by Phase\n"
                      "(PICRUSt2 MetaCyc pathways)")
    axes[1].set_ylabel("Shannon Entropy (nat log)")
    axes[1].set_xlabel("Mission Phase")
    axes[1].tick_params(axis='x', rotation=30)

    if resilience_ratio is not None:
        plt.suptitle(
            f"Functional Resilience Ratio = {resilience_ratio:.3f}\n"
            f"(Functional diversity changes "
            f"{100*resilience_ratio:.1f}% as much as taxonomic diversity)",
            fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "functional_redundancy_factor.png"),
                dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, "functional_redundancy_factor.pdf"),
                bbox_inches='tight')
    plt.close()
    print(f"Saved figure: results/figures/functional_redundancy_factor.png")

print("\nPipeline 29 complete.")
