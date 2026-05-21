# ==============================================================================
# Pipeline 30: Preprocessing Audit
# Produces a table of: (1) exact sequencing depth stats and exclusion threshold,
# (2) phase day boundaries, (3) software session info placeholder
# ==============================================================================

import os
import pandas as pd
import numpy as np

print("=" * 60)
print("Pipeline 30: Preprocessing Audit")
print("=" * 60)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
meta_dir = os.path.join(base_dir, "data", "metadata")
tab_dir  = os.path.join(base_dir, "results", "tables")

# -- 1. Sequencing depth audit --
counts = pd.read_csv(os.path.join(data_dir, "GLDS-191_GAmplicon_counts.tsv"),
                     sep="\t", index_col=0)
# counts: ASVs x samples -- transpose to samples x ASVs
counts_T = counts.T
read_depth = counts_T.sum(axis=1)

print(f"\n=== Sequencing Depth Stats (all samples before filtering) ===")
print(f"  Total samples:  {len(read_depth)}")
print(f"  Min depth:      {read_depth.min():.0f}")
print(f"  Max depth:      {read_depth.max():.0f}")
print(f"  Mean depth:     {read_depth.mean():.0f}")
print(f"  Median depth:   {read_depth.median():.0f}")
print(f"  Std depth:      {read_depth.std():.0f}")

# Common thresholds to audit
for threshold in [1000, 5000, 10000]:
    excluded = (read_depth < threshold).sum()
    print(f"  Samples < {threshold:,} reads: {excluded} excluded "
          f"({100*excluded/len(read_depth):.1f}%)")

# Identify the threshold used in pipeline 01
metadata = pd.read_csv(os.path.join(meta_dir, "processed_metadata.tsv"), sep="\t")
n_meta = len(metadata)
n_counts = len(read_depth)
print(f"\n  Samples in metadata: {n_meta}")
print(f"  Samples in counts:   {n_counts}")
if n_meta < n_counts:
    print(f"  => {n_counts - n_meta} samples were excluded during preprocessing")
    excluded_samples = set(read_depth.index) - set(metadata['SampleID'])
    exc_depths = read_depth[list(excluded_samples)]
    print(f"  Excluded sample depths: {sorted(exc_depths.values)}")
    print(f"  => Inferred threshold: < {exc_depths.max() + 1:.0f} reads")

# -- 2. Phase day boundary table --
print("\n=== Phase Day Boundaries ===")
phase_days = metadata.groupby('Phase')['Timepoint_Day'].agg(['min','max','count'])
phase_days = phase_days.reindex(
    ['Pre-Mission','Early','Mid','Late','Post-Mission'])
print(phase_days)

phase_boundaries = phase_days.reset_index().rename(columns={
    'Phase': 'Phase',
    'min': 'Day_min',
    'max': 'Day_max',
    'count': 'N_samples'
})
phase_boundaries['Biological_rationale'] = [
    'Pre-isolation baseline',
    'Acute adaptation (first 45 days)',
    'Steady-state confinement',
    'Deep chronic confinement',
    'Post-isolation recovery'
]
phase_boundaries.to_csv(
    os.path.join(tab_dir, "phase_boundary_table.csv"), index=False)
print(f"Saved to results/tables/phase_boundary_table.csv")

# -- 3. Depth distribution summary for Methods --
meta_samples = metadata['SampleID'].tolist()
valid_samples = [s for s in meta_samples if s in read_depth.index]

depth_summary = pd.DataFrame({
    'Statistic': ['Total samples (pre-filter)', 'Total samples (post-filter)',
                  'Samples excluded',
                  'Min depth (post-filter)',
                  'Max depth (post-filter)',
                  'Mean depth (post-filter)',
                  'Median depth (post-filter)'],
    'Value': [
        n_counts,
        n_meta,
        n_counts - n_meta,
        read_depth[valid_samples].min() if valid_samples else 'N/A',
        read_depth[valid_samples].max() if valid_samples else 'N/A',
        round(read_depth[valid_samples].mean(), 0) if valid_samples else 'N/A',
        round(read_depth[valid_samples].median(), 0) if valid_samples else 'N/A',
    ]
})
depth_summary.to_csv(
    os.path.join(tab_dir, "sequencing_depth_audit.csv"), index=False)
print("\n=== Depth Summary for Methods Section ===")
print(depth_summary.to_string(index=False))

print("\nPipeline 30 complete.")
