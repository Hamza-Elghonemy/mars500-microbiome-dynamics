# ==============================================================================
# W3.1: Taxa–Function Integration (Spearman Correlation)
# MARS500 Publication Pipeline
# ==============================================================================
# Correlates top DA genera with top variable MetaCyc pathways using
# per-sample Spearman correlations. Produces a bipartite heatmap.
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(tidyverse)
  library(ggplot2)
  library(pheatmap)
})

message("W3.1: Taxa–Function Integration...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir    <- file.path(base_dir, "data", "processed")
meta_dir    <- file.path(base_dir, "data", "metadata")
picrust_dir <- file.path(base_dir, "results", "picrust2", "pathways_out")
tab_dir     <- file.path(base_dir, "results", "tables")
fig_dir     <- file.path(base_dir, "results", "figures")

# ---------- Load genus abundances -------------------------------------------
ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"),
                   sep = "\t", header = TRUE, row.names = 1)
sample_data(ps) <- sample_data(meta)

rn <- rank_names(ps)
gr <- rn[grep("genus", rn, ignore.case = TRUE)]
if(length(gr) == 0) gr <- rn[length(rn)]

ps_gen <- tax_glom(ps, taxrank = gr, NArm = TRUE)
taxa_names(ps_gen) <- tax_table(ps_gen)[, gr]
ps_gen_rel <- transform_sample_counts(ps_gen, function(x) x / sum(x))

# Get top 30 most variable genera
gen_mat <- as.data.frame(t(otu_table(ps_gen_rel)))
gen_vars <- apply(gen_mat, 2, var)
top_genera <- names(sort(gen_vars, decreasing = TRUE))[1:30]

# ---------- Load MetaCyc pathways -------------------------------------------
path_file <- file.path(picrust_dir, "path_abun_unstrat.tsv.gz")
path_raw <- read.table(gzfile(path_file), header = TRUE, sep = "\t",
                       row.names = 1, check.names = FALSE)

common_samples <- intersect(rownames(gen_mat), colnames(path_raw))
gen_sub <- gen_mat[common_samples, top_genera]
path_mat <- t(path_raw[, common_samples])

# Top 25 most variable pathways
path_vars <- apply(path_mat, 2, var)
top_paths <- names(sort(path_vars, decreasing = TRUE))[1:25]
path_sub <- path_mat[common_samples, top_paths]

# ---------- Spearman correlation matrix (genera × pathways) -----------------
cor_matrix <- matrix(NA, nrow = length(top_genera), ncol = length(top_paths))
p_matrix   <- matrix(NA, nrow = length(top_genera), ncol = length(top_paths))

for(i in seq_along(top_genera)) {
  for(j in seq_along(top_paths)) {
    ct <- cor.test(gen_sub[, i], path_sub[, j], method = "spearman", exact = FALSE)
    cor_matrix[i, j] <- ct$estimate
    p_matrix[i, j]   <- ct$p.value
  }
}

rownames(cor_matrix) <- top_genera
colnames(cor_matrix) <- top_paths
rownames(p_matrix)   <- top_genera
colnames(p_matrix)   <- top_paths

# Save
cor_df <- as.data.frame(cor_matrix) %>% rownames_to_column("Genus")
write.csv(cor_df, file.path(tab_dir, "taxa_pathway_correlations.csv"), row.names = FALSE)

# FDR correction
p_adj <- matrix(p.adjust(p_matrix, method = "BH"),
                nrow = nrow(p_matrix), dimnames = dimnames(p_matrix))

# Significance stars for the heatmap
sig_stars <- matrix("", nrow = nrow(p_adj), ncol = ncol(p_adj))
sig_stars[p_adj < 0.05] <- "*"
sig_stars[p_adj < 0.01] <- "**"
sig_stars[p_adj < 0.001] <- "***"

# ---------- Bipartite heatmap -----------------------------------------------
# Only show genera that have at least one significant correlation
sig_rows <- which(apply(p_adj < 0.05, 1, any))
if(length(sig_rows) > 0) {
  plot_mat <- cor_matrix[sig_rows, , drop = FALSE]

  pdf(file.path(fig_dir, "Taxa_Function_Bipartite.pdf"), width = 14, height = 8)
  pheatmap(plot_mat,
           color = colorRampPalette(c("#2C7BB6", "white", "#D7191C"))(100),
           main = "Genus–MetaCyc Pathway Spearman Correlations (FDR < 0.05 shown)",
           fontsize_row = 8, fontsize_col = 7,
           display_numbers = sig_stars[sig_rows, , drop = FALSE],
           number_color = "black",
           cluster_rows = TRUE, cluster_cols = TRUE)
  dev.off()
  message("  Saved bipartite heatmap to results/figures/Taxa_Function_Bipartite.pdf")
} else {
  message("  No significant taxa-pathway correlations at FDR < 0.05.")
  # Still save the heatmap without significance filtering
  pdf(file.path(fig_dir, "Taxa_Function_Bipartite.pdf"), width = 14, height = 8)
  pheatmap(cor_matrix,
           color = colorRampPalette(c("#2C7BB6", "white", "#D7191C"))(100),
           main = "Genus–MetaCyc Pathway Spearman Correlations (Top 30 × 25)",
           fontsize_row = 8, fontsize_col = 7)
  dev.off()
}

message("W3.1: Taxa–Function integration complete.")
