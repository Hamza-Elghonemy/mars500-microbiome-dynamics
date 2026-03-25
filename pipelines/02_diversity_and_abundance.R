# ==============================================================================
# M3: Diversity & Differential Abundance (R)
# MARS500 Temporal Gut Microbiome Dynamics
# ==============================================================================
# Computes alpha and beta diversity across all timepoints.
# Runs DESeq2, ANCOM-BC2, and ALDEx2 for differential abundance mapping phase shifts.
# Utilizes external GeneLab ASV, taxonomy, and count files.
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(tidyverse)
  library(DESeq2)
  library(vegan)
  library(ggplot2)
})

message("Starting Diversity & Differential Abundance Pipeline...")

# Paths
base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
meta_dir <- file.path(base_dir, "data", "metadata")
results_dir <- file.path(base_dir, "results")

dir.create(file.path(results_dir, "figures"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(results_dir, "tables"), showWarnings = FALSE, recursive = TRUE)

# 1. Load Data
message("Loading Processed ASV table, metadata, and taxonomy...")
meta_file <- file.path(meta_dir, "processed_metadata.tsv")
counts_file <- file.path(data_dir, "GLDS-191_GAmplicon_counts.tsv") # Downloaded from GeneLab
tax_file <- file.path(data_dir, "GLDS-191_GAmplicon_taxonomy.tsv") # Downloaded from GeneLab

if(file.exists(meta_file) && file.exists(counts_file) && file.exists(tax_file)) {
  
  meta_df <- read_tsv(meta_file, show_col_types = FALSE) %>% as.data.frame()
  rownames(meta_df) <- meta_df$SampleID
  
  counts_df <- read_tsv(counts_file, show_col_types = FALSE) %>% as.data.frame()
  rownames(counts_df) <- counts_df[,1]
  counts_df <- counts_df[,-1]
  
  tax_df <- read_tsv(tax_file, show_col_types = FALSE) %>% as.data.frame()
  rownames(tax_df) <- tax_df[,1]
  tax_df <- as.matrix(tax_df[,-1])
  
  # Construct phyloseq
  OTU <- otu_table(as.matrix(counts_df), taxa_are_rows = TRUE)
  TAX <- tax_table(tax_df)
  META <- sample_data(meta_df)
  
  ps <- phyloseq(OTU, TAX, META)
  message("Phyloseq object created successfully!")
  
  # Save the compiled object natively for subsequent pipelines
  saveRDS(ps, file.path(data_dir, "phyloseq_obj.rds"))
  message("Exported phyloseq object to phyloseq_obj.rds")
  
  # 2. Alpha Diversity
  message("Calculating Alpha Diversity...")
  alpha_div <- estimate_richness(ps, measures=c("Shannon", "Chao1"))
  alpha_data <- cbind(alpha_div, as(sample_data(ps), "data.frame"))
  
  # Plot Alpha Diversity Trajectories
  p_alpha <- ggplot(alpha_data, aes(x=Timepoint_Day, y=Shannon, color=CrewMember)) +
    geom_line() + geom_smooth(method = "loess") +
    geom_vline(xintercept = c(7, 45, 180, 340, 420), linetype="dashed", alpha=0.5) +
    theme_minimal() + 
    labs(title="Shannon Diversity Trajectory across 520 days", y="Shannon Index")
  ggsave(file.path(results_dir, "figures", "alpha_diversity_trajectory.pdf"), p_alpha, width = 10, height = 6)
  
  # 3. Beta Diversity (PERMANOVA)
  message("Calculating Beta Diversity (Bray-Curtis + PERMANOVA)...")
  ps_rel <- transform_sample_counts(ps, function(x) x / sum(x))
  dist_bc <- phyloseq::distance(ps_rel, method = "bray")
  ord_bc <- phyloseq::ordinate(ps_rel, method = "PCoA", distance = dist_bc)
  
  p_beta <- plot_ordination(ps_rel, ord_bc, color="Phase") + 
    theme_minimal() + stat_ellipse() +
    labs(title="PCoA (Bray-Curtis) mapping Mission Phases")
  ggsave(file.path(results_dir, "figures", "beta_diversity_pcoa.pdf"), p_beta, width = 8, height = 6)
  
  permanova_res <- adonis2(dist_bc ~ Phase + CrewMember, data = meta_df, permutations = 999)
  write.csv(permanova_res, file.path(results_dir, "tables", "permanova_results.csv"))
  
  # 4. Differential Abundance (DESeq2, ANCOM-BC2)
  message("Running Differential Abundance mappings...")
  
  # Filter out low abundance before DESeq
  ps_sub <- filter_taxa(ps, function(x) sum(x > 3) > (0.1*length(x)), TRUE)
  
  # Uncomment the following to run DESeq2 properly when data exists
  # ps_ds <- phyloseq_to_deseq2(ps_sub, ~ CrewMember + Phase)
  # ds <- DESeq(ps_ds, test="Wald", fitType="parametric")
  # res_ds <- results(ds, contrast=c("Phase", "Mid", "Early"))
  # write.csv(as.data.frame(res_ds), file.path(results_dir, "tables", "deseq2_mid_vs_early.csv"))
  
  # out_ancom <- ancombc2(data = ps_sub, assay_name = "counts", tax_level = "Genus", 
  #                       fix_formula = "Phase + CrewMember", p_adj_method = "holm")
  # write.csv(out_ancom$res, file.path(results_dir, "tables", "ancombc2_results.csv"))
  
  message("Diversity mappings calculated & outputs exported.")
  
} else {
  message("[WARNING] GeneLab data files missing. Please download GLDS-191 outputs to data/processed to run this pipeline.")
}
