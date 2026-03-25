# MARS500 Temporal Gut Microbiome Dynamics

A comprehensive re-analysis of the MARS500 520-day simulated Mars mission gut microbiome dataset (NASA GeneLab GLDS-191 / OSD-191). This project applies modern bioinformatics, network ecology, and machine learning to reveal temporal microbiome dynamics during prolonged human isolation — spanning taxonomy, function, network topology, and temporal biomarker discovery.

> **Associated Dataset:** [GLDS-191 on NASA OSDR](https://osdr.nasa.gov/bio/repo/data/studies/OSD-191)

---

## Key Findings

| Analysis                                       | Result                                                                      |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| **PERMANOVA** (Taxonomic)                | Phase R² = 0.055, p = 0.001; strongest contrast: Early vs Late             |
| **PERMANOVA** (Functional)               | Phase R² = 0.042, p = 0.002                                                |
| **DESeq2** (Late vs Pre-Mission)         | *Sutterella* and *Bilophila* enriched; *Faecalibacterium* depleted    |
| **ANCOM-BC2**                            | Conservative validation; 0 genera at FDR < 0.05 (Pre-Mission n = 4)         |
| **Random Forest** (Phase Classification) | CV Accuracy = 0.616, permutation p = 0.001 (1000 perms)                     |
| **Temporal Regression**                  | RF R² = 0.63; microbiome predicts isolation day within ~68-day MAE         |
| **Network Disintegration**               | Progressive density loss from Pre-Mission to Late confinement               |
| **Keystone Permutation Test**            | Hub taxa 25–611σ above null (p < 0.001, all phases)                       |
| **Longitudinal GAM**                     | Shannon diversity: EDF = 2.15, p = 1.07 × 10⁻⁶, 44.9% deviance explained |

---

## Repository Structure

```
research/
├── data/
│   ├── metadata/            # Processed sample metadata (phase assignments)
│   ├── processed/           # GeneLab ASV counts, taxonomy, phyloseq objects
│   ├── raw/                 # (Optional) Raw FASTQ files
│   └── reference/           # Original ISA metadata from GeneLab
├── pipelines/               # Reproducible analysis scripts (see below)
├── results/
│   ├── figures/             # All related graphs 
│   ├── tables/              # Statistical result tables (CSV)
│   ├── networks/            # Serialized igraph network objects (RDS)
│   ├── models/              # ML model outputs and feature importances
│   └── picrust2/            # PICRUSt2 MetaCyc pathway predictions
└── README.md
```

---

## Pipelines

All scripts are located in `pipelines/` and are designed to run sequentially. Each script auto-detects its working directory relative to the project root.

### Phase I: Data Preparation

| Script                           | Language | Description                                                         |
| -------------------------------- | -------- | ------------------------------------------------------------------- |
| `01_data_preprocessing.py`     | Python   | Parses GeneLab ISA metadata; assigns 5 mission phases               |
| `02_diversity_and_abundance.R` | R        | Constructs phyloseq object; alpha/beta diversity; initial PERMANOVA |

### Phase II: Core Analyses

| Script                          | Language | Description                                                              |
| ------------------------------- | -------- | ------------------------------------------------------------------------ |
| `03_machine_learning.py`      | Python   | Random Forest and XGBoost phase classification with cross-validation     |
| `04_network_analysis.R`       | R        | Spearman co-occurrence networks per mission phase (ρ > 0.6 threshold)   |
| `05_functional_prediction.py` | Python   | PICRUSt2 wrapper for MetaCyc pathway inference                           |
| `06_differential_abundance.R` | R        | DESeq2 differential abundance (Late vs Pre-Mission) with volcano plots   |
| `07_network_visualization.R`  | R        | Network topology metrics, keystone identification, and visualization     |
| `08_picrust2_conda_run.sh`    | Bash     | Automated PICRUSt2 conda environment setup (Rosetta 2 for Apple Silicon) |

### Phase III: Visualization and Synthesis

| Script                                | Language | Description                                                                |
| ------------------------------------- | -------- | -------------------------------------------------------------------------- |
| `09_comprehensive_visualizations.R` | R        | Multi-panel publication figures (heatmaps, functional PCoA, network decay) |
| `10_temporal_ml.py`                 | Python   | Temporal regression (Random Forest + XGBoost) predicting isolation day     |

### Phase IV: Statistical Rigor and Publication Analyses

| Script                             | Language | Description                                                                  |
| ---------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `11_permanova_pairwise.R`        | R        | Global + pairwise PERMANOVA (taxonomic and functional) with FDR correction   |
| `12_ancombc_validation.R`        | R        | ANCOM-BC2 compositionality-aware differential abundance validation           |
| `13_rf_permutation_test.py`      | Python   | 1000-iteration permutation test on RF classifier; null distribution plot     |
| `14_metadata_correlations.R`     | R        | Phase-level taxa/network correlations with literature-derived physiology     |
| `15_taxa_function_integration.R` | R        | Genus-to-MetaCyc pathway Spearman correlations; bipartite heatmap            |
| `16_keystone_statistics.R`       | R        | Formal centrality metrics with 1000-iteration permutation significance tests |
| `17_longitudinal_gam.R`          | R        | GAM spline models for Shannon, Simpson, and 6 biomarker genera trajectories  |

---

## Requirements

### R (>= 4.3)

```
phyloseq, vegan, DESeq2, ANCOMBC, igraph, mgcv, lme4,
tidyverse, ggplot2, ggrepel, pheatmap, microbiome
```

### Python (>= 3.9)

```
pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
```

### PICRUSt2 (Conda)

PICRUSt2 requires an isolated conda environment. On Apple Silicon, Rosetta 2 emulation is needed:

```bash
CONDA_SUBDIR=osx-64 conda create -y -n picrust2_env \
    -c bioconda -c conda-forge picrust2=2.5.2
conda run -n picrust2_env conda config --env --set subdir osx-64
```

Alternatively, run `bash pipelines/08_picrust2_conda_run.sh` to auto-configure the environment.

---

## Quick Start

1. **Download data** from [NASA GeneLab GLDS-191](https://osdr.nasa.gov/bio/repo/data/studies/OSD-191). Place the processed outputs into `data/processed/`:

   - `GLDS-191_GAmplicon_counts.tsv`
   - `GLDS-191_GAmplicon_taxonomy.tsv`
   - `GLDS-191_GAmplicon_ASVs.fasta`
2. **Run the preprocessing pipeline:**

   ```bash
   python pipelines/01_data_preprocessing.py
   ```
3. **Execute pipelines sequentially:**

   ```bash
   Rscript pipelines/02_diversity_and_abundance.R
   python pipelines/03_machine_learning.py
   Rscript pipelines/04_network_analysis.R
   # ... continue through pipeline 17
   ```
4. **Check outputs** in `results/figures/` (PDF) and `results/tables/` (CSV).

---

## Citation

If you use this code or data in your research, please cite:

- **Original dataset:** Turroni, S. et al. (2017). Temporal dynamics of the gut microbiota in people sharing a confined environment. *PLOS ONE.* [DOI:10.1371/journal.pone.0176991](https://doi.org/10.1371/journal.pone.0176991)
- **Reanalysis reference:** Brereton, N.J.B. & Gonzalez, E. (2021). Reanalysis of the Mars500 experiment reveals common gut microbiome alterations. *Computational and Structural Biotechnology Journal.* [DOI:10.1016/j.csbj.2020.12.040](https://doi.org/10.1016/j.csbj.2020.12.040)

---

## License

This project is provided for academic and research purposes. The MARS500 microbiome data is publicly available through [NASA&#39;s Open Science Data Repository](https://osdr.nasa.gov/).
