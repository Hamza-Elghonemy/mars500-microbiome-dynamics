# ==============================================================================
# M5: Network Analysis (R)
# MARS500 Temporal Gut Microbiome Dynamics
# ==============================================================================
# Spearman co-occurrence network analysis (per mission phase).
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(tidyverse)
  library(igraph)
})

message("Starting Spearman Network Inference Pipeline...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
meta_dir <- file.path(base_dir, "data", "metadata")
networks_dir <- file.path(base_dir, "results", "networks")
dir.create(networks_dir, showWarnings = FALSE, recursive = TRUE)

# 1. Load Data
counts_file <- file.path(data_dir, "GLDS-191_GAmplicon_counts.tsv")
meta_file   <- file.path(meta_dir, "processed_metadata.tsv")
tax_file    <- file.path(data_dir, "GLDS-191_GAmplicon_taxonomy.tsv")

if(file.exists(counts_file) && file.exists(meta_file) && file.exists(tax_file)) {
    ps_rds <- file.path(data_dir, "phyloseq_obj.rds")
    if (file.exists(ps_rds)) {
      ps <- readRDS(ps_rds)
      
      message("Creating separate networks per phase...")
      phases <- unique(sample_data(ps)$Phase)
      
      for (p in phases) {
        if(is.na(p)) next
        cat("Inferring Spearman Network for Phase:", p, "\n")
        
        ps_sub <- subset_samples(ps, Phase == p)
        # Filter sparse taxa
        ps_sub <- filter_taxa(ps_sub, function(x) sum(x > 0) > (0.1*length(x)), TRUE)
        
        # Calculate Spearman correlation matrix
        otu_mat <- t(as(otu_table(ps_sub), "matrix"))
        cor_mat <- cor(otu_mat, method = "spearman")
        
        # Threshold (rho > 0.6)
        cor_mat[cor_mat < 0.6 & cor_mat > -0.6] <- 0
        diag(cor_mat) <- 0
        
        ig_p <- graph_from_adjacency_matrix(cor_mat, mode = "undirected", weighted = TRUE)
        saveRDS(ig_p, file.path(networks_dir, paste0("spearman_network_", p, ".rds")))
      }
      message("Spearman Networks constructed successfully.")
    } else {
      message("[WARNING] Phyloseq RDS object missing. Run Pipeline 02 first.")
    }
} else {
  message("[WARNING] GeneLab data files missing.")
}
