# ==============================================================================
# Pipeline 26: MaAsLin2 Validation
# Third differential abundance method to assess DESeq2 concordance
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(Maaslin2)
  library(tidyverse)
})

message("Starting MaAsLin2 Validation Pipeline...")

base_dir <- getwd()
if (basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir    <- file.path(base_dir, "data", "processed")
results_dir <- file.path(base_dir, "results", "tables")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
ps_genus <- tax_glom(ps, taxrank = "genus", NArm = FALSE)
ps_genus <- filter_taxa(ps_genus, function(x) sum(x > 3) > (0.1 * length(x)), TRUE)

# Keep only Late and Pre-Mission samples (matching DESeq2 comparison)
ps_sub <- subset_samples(ps_genus, Phase %in% c("Pre-Mission", "Late"))

# Extract OTU matrix (samples x features) as plain numeric data.frame
otu_raw  <- as(otu_table(ps_sub), "matrix")
# taxa_are_rows = TRUE in this phyloseq object, so transpose
otu_mat  <- as.data.frame(t(otu_raw))

# Get metadata and taxonomy
meta_raw <- as(sample_data(ps_sub), "data.frame")
taxa_map <- as.data.frame(tax_table(ps_sub))

# Rename OTU columns to genus names (keep ASV ID for NA genera)
genus_names <- taxa_map[colnames(otu_mat), "genus"]
genus_names[is.na(genus_names)] <- colnames(otu_mat)[is.na(genus_names)]
# Make syntactically valid and unique names (MaAsLin2 sanitizes anyway)
genus_names <- make.unique(make.names(genus_names))
colnames(otu_mat) <- genus_names

# Build a clean plain data.frame for metadata
meta_clean <- data.frame(
  Phase      = as.character(meta_raw$Phase),
  CrewMember = as.character(meta_raw$CrewMember),
  stringsAsFactors = FALSE
)
rownames(meta_clean) <- rownames(meta_raw)
# Set reference level
meta_clean$Phase <- factor(meta_clean$Phase, levels = c("Pre-Mission", "Late"))

maaslin2_out <- file.path(base_dir, "results", "maaslin2_output")

# Force all OTU columns to numeric
for (col in colnames(otu_mat)) {
  otu_mat[[col]] <- as.numeric(otu_mat[[col]])
}

fit <- Maaslin2(
  input_data      = otu_mat,
  input_metadata  = meta_clean,
  output          = maaslin2_out,
  fixed_effects   = c("Phase"),
  random_effects  = c("CrewMember"),
  normalization   = "CLR",
  transform       = "NONE",
  analysis_method = "LM",
  min_prevalence  = 0.20,
  correction      = "BH",
  plot_heatmap    = FALSE,
  plot_scatter    = FALSE
)

# Extract significant results
sig <- fit$results %>%
  filter(qval < 0.05) %>%
  arrange(qval)

message(sprintf("MaAsLin2 found %d significant genera (q < 0.05)", nrow(sig)))

# Load DESeq2 results for concordance table
deseq2 <- read.csv(file.path(results_dir, "DESeq2_Late_vs_PreMission.csv"),
                   row.names = 1) %>%
  filter(!is.na(padj)) %>%
  mutate(deseq2_sig   = padj < 0.05,
         deseq2_dir   = sign(log2FoldChange))

# MaAsLin2 sanitizes feature names with make.names, so we need to match
# Map sanitized genus names back to original genus names for merging
genus_lookup <- data.frame(
  original = taxa_map$genus,
  sanitized = make.unique(make.names(
    ifelse(is.na(taxa_map$genus), rownames(taxa_map), taxa_map$genus)
  )),
  stringsAsFactors = FALSE
)

maaslin_clean <- fit$results %>%
  select(feature, coef, qval) %>%
  left_join(genus_lookup, by = c("feature" = "sanitized")) %>%
  mutate(Genus = ifelse(is.na(original), feature, original)) %>%
  select(Genus, coef, qval) %>%
  rename(maaslin2_coef = coef, maaslin2_qval = qval) %>%
  mutate(maaslin2_sig = maaslin2_qval < 0.05,
         maaslin2_dir = sign(maaslin2_coef))

# Merge on genus name
concordance <- deseq2 %>%
  left_join(maaslin_clean, by = c("Genus" = "Genus")) %>%
  mutate(
    Evidence_Tier = case_when(
      deseq2_sig & maaslin2_sig & deseq2_dir == maaslin2_dir ~ "Replicated",
      deseq2_sig & !is.na(maaslin2_sig) & deseq2_dir == maaslin2_dir ~ "Exploratory",
      deseq2_sig ~ "Unconfirmed",
      TRUE ~ "Not significant"
    )
  ) %>%
  select(Genus, log2FoldChange, padj, maaslin2_coef, maaslin2_qval, Evidence_Tier) %>%
  arrange(Evidence_Tier, padj)

write.csv(concordance,
          file.path(results_dir, "three_method_DA_concordance.csv"),
          row.names = FALSE)

# Print summary
message("\n=== Three-Method Concordance Summary ===")
print(table(concordance$Evidence_Tier))

# Zero-count profile for Sutterella and Bilophila in Pre-Mission samples
ps_pre <- subset_samples(ps_genus, Phase == "Pre-Mission")
for (genus_name in c("Sutterella", "Bilophila")) {
  asv_ids <- rownames(taxa_map)[taxa_map$genus == genus_name & !is.na(taxa_map$genus)]
  if (length(asv_ids) > 0) {
    counts_pre <- as.matrix(otu_table(ps_pre))[asv_ids, , drop = FALSE]
    n_samples  <- ncol(counts_pre)
    n_zero     <- sum(colSums(counts_pre) == 0)
    message(sprintf("%s: %d/%d Pre-Mission samples are zero (%.1f%%)",
                    genus_name, n_zero, n_samples, 100 * n_zero / n_samples))
  }
}

message("MaAsLin2 validation complete.")
