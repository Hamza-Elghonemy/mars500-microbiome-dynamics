# ==============================================================================
# W1.1–W1.2: PERMANOVA Pairwise Contrasts (Taxonomic + Functional)
# MARS500 Publication Pipeline
# ==============================================================================
# Runs adonis2 on Bray-Curtis distances for both ASV-level taxonomy and
# PICRUSt2 MetaCyc functional pathways. Reports global test + all pairwise
# phase contrasts with FDR correction.
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(vegan)
  library(tidyverse)
})

message("W1.1: PERMANOVA / Adonis2 — Taxonomic & Functional...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir   <- file.path(base_dir, "data", "processed")
meta_dir   <- file.path(base_dir, "data", "metadata")
picrust_dir <- file.path(base_dir, "results", "picrust2", "pathways_out")
tab_dir    <- file.path(base_dir, "results", "tables")
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

# ---------- Helper: manual pairwise PERMANOVA --------------------------------
pairwise_permanova <- function(dist_mat, grouping, n_perm = 999) {
  levels <- levels(grouping)
  combos <- combn(levels, 2)
  results <- data.frame()

  for(i in 1:ncol(combos)) {
    pair <- combos[, i]
    idx  <- which(grouping %in% pair)
    sub_dist <- as.dist(as.matrix(dist_mat)[idx, idx])
    sub_grp  <- droplevels(grouping[idx])
    ad <- adonis2(sub_dist ~ sub_grp, permutations = n_perm)
    results <- rbind(results, data.frame(
      Pair      = paste(pair, collapse = " vs "),
      R2        = ad$R2[1],
      F_stat    = ad$F[1],
      p_value   = ad$`Pr(>F)`[1]
    ))
  }
  results$p_adj <- p.adjust(results$p_value, method = "BH")
  return(results)
}

# ---------- Load metadata ----------------------------------------------------
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"),
                   sep = "\t", header = TRUE, row.names = 1)
phase_order <- c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission")
meta$Phase <- factor(meta$Phase, levels = phase_order)

# =============================================================================
# PART A: TAXONOMIC PERMANOVA
# =============================================================================
message("  → Taxonomic Bray-Curtis PERMANOVA...")
ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
sample_data(ps) <- sample_data(meta)

# Relative abundance
ps_rel <- transform_sample_counts(ps, function(x) x / sum(x))
bc_tax <- phyloseq::distance(ps_rel, method = "bray")

# Global adonis2 (Phase + CrewMember)
global_tax <- adonis2(bc_tax ~ Phase + CrewMember, data = as(sample_data(ps_rel), "data.frame"),
                      permutations = 999, by = "margin")
message("  Taxonomic PERMANOVA (global):")
print(global_tax)

# Pairwise
pw_tax <- pairwise_permanova(bc_tax, sample_data(ps_rel)$Phase)
message("  Pairwise taxonomic contrasts:")
print(pw_tax)

# Save
write.csv(as.data.frame(global_tax), file.path(tab_dir, "permanova_taxonomic_global.csv"))
write.csv(pw_tax, file.path(tab_dir, "permanova_taxonomic_pairwise.csv"), row.names = FALSE)

# =============================================================================
# PART B: FUNCTIONAL (MetaCyc) PERMANOVA
# =============================================================================
message("  → Functional MetaCyc Bray-Curtis PERMANOVA...")
path_file <- file.path(picrust_dir, "path_abun_unstrat.tsv.gz")

if(file.exists(path_file)) {
  path_raw <- read.table(gzfile(path_file), header = TRUE, sep = "\t",
                         row.names = 1, check.names = FALSE)
  common   <- intersect(colnames(path_raw), rownames(meta))
  path_mat <- path_raw[, common]
  meta_f   <- meta[common, ]

  path_rel <- sweep(path_mat, 2, colSums(path_mat), "/")
  bc_func  <- vegdist(t(path_rel), method = "bray")

  global_func <- adonis2(bc_func ~ Phase + CrewMember, data = meta_f,
                         permutations = 999, by = "margin")
  message("  Functional PERMANOVA (global):")
  print(global_func)

  pw_func <- pairwise_permanova(bc_func, meta_f$Phase)
  message("  Pairwise functional contrasts:")
  print(pw_func)

  write.csv(as.data.frame(global_func), file.path(tab_dir, "permanova_functional_global.csv"))
  write.csv(pw_func, file.path(tab_dir, "permanova_functional_pairwise.csv"), row.names = FALSE)

} else {
  message("[WARNING]  PICRUSt2 pathway file not found. Skipping functional PERMANOVA.")
}

message("W1.1–W1.2: PERMANOVA complete. Results in results/tables/")
