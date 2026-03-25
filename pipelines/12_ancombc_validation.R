# ==============================================================================
# W1.3: ANCOM-BC2 Validation of Differential Abundance
# MARS500 Publication Pipeline
# ==============================================================================
# Complements the existing DESeq2 analysis with a bias-corrected compositionality-
# aware method. Tests at Genus level across all 5 phases, with Late vs Pre-Mission
# as the primary contrast.
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(ANCOMBC)
  library(tidyverse)
  library(ggplot2)
})

message("W1.3: ANCOM-BC2 Differential Abundance Validation...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
meta_dir <- file.path(base_dir, "data", "metadata")
tab_dir  <- file.path(base_dir, "results", "tables")
fig_dir  <- file.path(base_dir, "results", "figures")

# Load phyloseq
ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"),
                   sep = "\t", header = TRUE, row.names = 1)
phase_order <- c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission")
meta$Phase <- factor(meta$Phase, levels = phase_order)
sample_data(ps) <- sample_data(meta)

# Agglomerate to Genus
rank_names_ps <- rank_names(ps)
genus_rank <- rank_names_ps[grep("genus", rank_names_ps, ignore.case = TRUE)]
if(length(genus_rank) == 0) genus_rank <- rank_names_ps[length(rank_names_ps)]

ps_gen <- tax_glom(ps, taxrank = genus_rank, NArm = TRUE)

# Run ANCOM-BC2 with Phase as fixed effect
message("  Running ANCOM-BC2 (this may take a few minutes)...")
out <- ancombc2(
  data       = ps_gen,
  fix_formula = "Phase",
  p_adj_method = "BH",
  prv_cut    = 0.10,  # Prevalence filter: present in ≥10% of samples
  group      = "Phase",
  struc_zero = TRUE,
  neg_lb     = TRUE,
  verbose    = TRUE
)

# Extract results — ANCOM-BC2 v2.12+ returns a single flat data.frame in out$res
res <- out$res

# Build a summary table for Late vs Pre-Mission
late_lfc_col  <- "lfc_PhaseLate"
late_q_col    <- "q_PhaseLate"
late_diff_col <- "diff_PhaseLate"
late_robust_col <- "diff_robust_PhaseLate"

if(all(c(late_lfc_col, late_q_col, late_diff_col) %in% colnames(res))) {
  # Get the taxon names
  taxa_ids <- res$taxon
  tax_tab  <- as.data.frame(tax_table(ps_gen))

  summary_df <- data.frame(
    Taxon       = taxa_ids,
    Genus       = tax_tab[taxa_ids, genus_rank],
    LFC         = res[[late_lfc_col]],
    q_value     = res[[late_q_col]],
    Significant = res[[late_diff_col]],
    Robust      = if(late_robust_col %in% colnames(res)) res[[late_robust_col]] else NA
  )

  summary_df <- summary_df %>% arrange(desc(abs(LFC)))

  write.csv(summary_df, file.path(tab_dir, "ANCOMBC_Late_vs_PreMission.csv"), row.names = FALSE)
  message("  Saved ANCOM-BC results to results/tables/ANCOMBC_Late_vs_PreMission.csv")

  # Plot top significant genera
  sig_df <- summary_df %>% filter(Significant == TRUE) %>%
    slice_max(abs(LFC), n = 20) %>%
    mutate(Direction = ifelse(LFC > 0, "Enriched in Late", "Depleted in Late"))

  if(nrow(sig_df) > 0) {
    p <- ggplot(sig_df, aes(x = reorder(Genus, LFC), y = LFC, fill = Direction)) +
      geom_col() +
      coord_flip() +
      scale_fill_manual(values = c("Enriched in Late" = "#D7191C", "Depleted in Late" = "#2C7BB6")) +
      theme_minimal(base_size = 12) +
      labs(title = "ANCOM-BC2: Late vs Pre-Mission",
           subtitle = "Bias-corrected log-fold changes (FDR < 0.05)",
           x = "", y = "Log Fold Change") +
      theme(legend.position = "bottom")

    ggsave(file.path(fig_dir, "ANCOMBC_LFC_barplot.pdf"), p, width = 8, height = 7)
    message("  Saved ANCOM-BC barplot to results/figures/ANCOMBC_LFC_barplot.pdf")
  } else {
    message("  No significant genera at FDR < 0.05 for Late vs Pre-Mission.")
  }
} else {
  message("[WARNING]  Could not find 'PhaseLate' contrast columns in ANCOM-BC output.")
  message("  Available columns: ", paste(colnames(res), collapse = ", "))
}

# Save the full multi-phase results
write.csv(res, file.path(tab_dir, "ANCOMBC_all_phases.csv"), row.names = FALSE)

message("W1.3: ANCOM-BC2 validation complete.")
