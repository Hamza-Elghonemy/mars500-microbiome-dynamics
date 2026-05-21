# ==============================================================================
# Pipeline 27: Network Threshold Sensitivity Analysis
# Tests whether density tightening (Early → Late) is robust across
# Spearman correlation thresholds of 0.5, 0.6, 0.7, and 0.8
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(tidyverse)
  library(igraph)
})

message("Starting Network Threshold Sensitivity Analysis...")

base_dir <- getwd()
if (basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir    <- file.path(base_dir, "data", "processed")
results_dir <- file.path(base_dir, "results", "tables")

ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))

# Agglomerate to genus level (matching network pipeline 04)
ps_genus <- tax_glom(ps, taxrank = "genus", NArm = FALSE)

thresholds <- c(0.5, 0.6, 0.7, 0.8)
phases     <- c("Early", "Mid", "Late", "Post-Mission")
results    <- list()

for (rho_thresh in thresholds) {
  for (phase in phases) {
    ps_sub <- subset_samples(ps_genus, Phase == phase)

    # Minimum 20% prevalence filter (matching pipeline 04)
    min_prev <- 0.20 * nsamples(ps_sub)
    ps_sub   <- filter_taxa(ps_sub, function(x) sum(x > 0) >= min_prev, TRUE)

    n_samples <- nsamples(ps_sub)
    if (n_samples < 5) {
      message(sprintf("Skipping %s at rho=%.1f (only %d samples)", phase, rho_thresh, n_samples))
      next
    }

    otu_mat <- t(as(otu_table(ps_sub), "matrix"))

    # Compute empirical distribution of all pairwise rho values
    cor_mat <- cor(otu_mat, method = "spearman", use = "pairwise.complete.obs")
    all_rho <- cor_mat[upper.tri(cor_mat)]

    # Threshold
    adj_mat <- cor_mat
    adj_mat[adj_mat < rho_thresh & adj_mat > -rho_thresh] <- 0
    diag(adj_mat) <- 0

    g <- graph_from_adjacency_matrix(abs(adj_mat) > 0,
                                     mode = "undirected",
                                     diag = FALSE)

    N <- vcount(g)
    E <- ecount(g)
    D <- if (N > 1) 2 * E / (N * (N - 1)) else 0

    results[[length(results) + 1]] <- tibble(
      Phase          = phase,
      Threshold      = rho_thresh,
      N_Nodes        = N,
      N_Edges        = E,
      Density        = round(D, 4),
      N_Samples      = n_samples,
      Median_rho_all = round(median(abs(all_rho)), 4),
      Pct_above_thresh = round(mean(abs(all_rho) >= rho_thresh) * 100, 2)
    )

    message(sprintf("Phase: %-15s | rho > %.1f | N=%3d | E=%5d | D=%.3f",
                    phase, rho_thresh, N, E, D))
  }
}

sensitivity_df <- bind_rows(results)
write.csv(sensitivity_df,
          file.path(results_dir, "network_threshold_sensitivity.csv"),
          row.names = FALSE)

# Print tightening check: does D increase Early → Late at every threshold?
message("\n=== TIGHTENING CHECK (Early → Late density by threshold) ===")
check <- sensitivity_df %>%
  filter(Phase %in% c("Early", "Late")) %>%
  select(Threshold, Phase, Density) %>%
  pivot_wider(names_from = Phase, values_from = Density) %>%
  mutate(Tightening = Late > Early,
         Delta = round(Late - Early, 4))
print(check)

write.csv(check,
          file.path(results_dir, "network_tightening_check.csv"),
          row.names = FALSE)

message("Sensitivity analysis complete.")
