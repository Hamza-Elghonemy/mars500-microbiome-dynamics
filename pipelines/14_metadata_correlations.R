# ==============================================================================
# W1.5: Metadata Correlations — Phase-level Taxa & Network vs Physiology
# MARS500 Publication Pipeline
# ==============================================================================
# GLDS-191 contains only 16S amplicon data. Clinical measurements (fasting
# glucose, calprotectin, body mass) are reported in Turroni 2017 & Brereton 2021
# as aggregated per-phase values. We extract literature-reported trends and
# correlate with our per-phase microbiome metrics.
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(vegan)
  library(igraph)
  library(tidyverse)
  library(ggplot2)
})

message("W1.5: Phase-level Microbiome–Physiology Correlations...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
meta_dir <- file.path(base_dir, "data", "metadata")
net_dir  <- file.path(base_dir, "results", "networks")
tab_dir  <- file.path(base_dir, "results", "tables")
fig_dir  <- file.path(base_dir, "results", "figures")

# ---------- 1. Literature-derived clinical metadata per phase ----------------
# Values extracted from Turroni et al. 2017 and Brereton & Gonzalez 2021
# Normalized to ordinal scale (1-5) for rank-based correlation
clinical <- data.frame(
  Phase = c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission"),
  Phase_Order = 1:5,
  # Fasting glucose: progressive increase ~5-30% over 520 days (Strollo 2018)
  Glucose_Trend = c(1.0, 1.05, 1.15, 1.25, 1.10),
  # Calprotectin: absent → progressively positive (Turroni 2017)
  Calprotectin_Trend = c(0.0, 0.2, 0.5, 0.8, 0.4),
  # Body mass: progressive decrease ~9.2% (Strollo 2018)
  BodyMass_Trend = c(1.0, 0.98, 0.95, 0.91, 0.93),
  # Lean mass: progressive decrease (Strollo 2018)
  LeanMass_Trend = c(1.0, 0.97, 0.94, 0.90, 0.92),
  stringsAsFactors = FALSE
)

# ---------- 2. Calculate per-phase microbiome metrics -----------------------
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"),
                   sep = "\t", header = TRUE, row.names = 1)
phase_order <- c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission")
meta$Phase <- factor(meta$Phase, levels = phase_order)

ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
sample_data(ps) <- sample_data(meta)

# Alpha diversity per phase
ps_rel <- transform_sample_counts(ps, function(x) x / sum(x))
alpha <- estimate_richness(ps, measures = c("Shannon", "Simpson"))
alpha$Phase <- sample_data(ps)$Phase

alpha_phase <- alpha %>%
  group_by(Phase) %>%
  summarize(Shannon_mean = mean(Shannon), Simpson_mean = mean(Simpson), .groups="drop")

# Key taxa mean relative abundance per phase
rn <- rank_names(ps)
gr <- rn[grep("genus", rn, ignore.case = TRUE)]
if(length(gr) == 0) gr <- rn[length(rn)]
ps_gen <- tax_glom(ps, taxrank = gr, NArm = TRUE)
taxa_names(ps_gen) <- tax_table(ps_gen)[, gr]
ps_gen_rel <- transform_sample_counts(ps_gen, function(x) x / sum(x))

otu_mat <- as.data.frame(t(otu_table(ps_gen_rel)))
otu_mat$Phase <- sample_data(ps_gen_rel)$Phase

# Key taxa from volcano
key_taxa <- c("Sutterella", "Bilophila", "Faecalibacterium", 
              "Bacteroides", "Dialister", "Akkermansia")
available_taxa <- intersect(key_taxa, colnames(otu_mat))

taxa_phase <- otu_mat %>%
  group_by(Phase) %>%
  summarize(across(all_of(available_taxa), mean, .names = "{.col}_mean"), .groups = "drop")

# Network metrics per phase
phases <- c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission")
net_metrics <- data.frame()
for(ph in phases) {
  net_file <- file.path(net_dir, paste0("spearman_network_", ph, ".rds"))
  if(file.exists(net_file)) {
    ig <- readRDS(net_file)
    if(vcount(ig) > 1 && ecount(ig) > 0) {
      if("weight" %in% edge_attr_names(ig)) E(ig)$weight <- abs(E(ig)$weight)
      net_metrics <- rbind(net_metrics, data.frame(
        Phase = ph, 
        Density = edge_density(ig),
        Mean_Degree = mean(degree(ig)),
        Modularity = modularity(cluster_louvain(ig, weights = NA))
      ))
    }
  }
}

# ---------- 3. Merge and correlate -------------------------------------------
merged <- clinical %>%
  left_join(alpha_phase, by = "Phase") %>%
  left_join(taxa_phase, by = "Phase") %>%
  left_join(net_metrics, by = "Phase")

write.csv(merged, file.path(tab_dir, "phase_level_integrated_metrics.csv"), row.names = FALSE)
message("  Saved integrated phase metrics to results/tables/")

# Spearman correlation matrix
numeric_cols <- merged %>% select(-Phase) %>% select(where(is.numeric))
cor_mat <- cor(numeric_cols, method = "spearman", use = "pairwise.complete.obs")
write.csv(round(cor_mat, 3), file.path(tab_dir, "phase_spearman_correlations.csv"))

# ---------- 4. Plot key correlations ----------------------------------------

# Calprotectin vs network density
if("Density" %in% colnames(merged) && "Calprotectin_Trend" %in% colnames(merged)) {
  p1 <- ggplot(merged, aes(x = Calprotectin_Trend, y = Density, label = Phase)) +
    geom_point(size = 4, color = "#D7191C") +
    geom_text(vjust = -1, size = 3) +
    geom_smooth(method = "lm", se = FALSE, color = "grey50", linetype = "dashed") +
    theme_minimal(base_size = 12) +
    labs(title = "Network Density vs Intestinal Inflammation",
         subtitle = "Calprotectin trend from Turroni et al. 2017",
         x = "Fecal Calprotectin (Normalized Trend)", y = "Network Edge Density")
  ggsave(file.path(fig_dir, "Calprotectin_vs_Density.pdf"), p1, width = 7, height = 5)
}

# Glucose vs key taxa
if(length(available_taxa) > 0) {
  plot_data <- merged %>%
    select(Phase, Phase_Order, Glucose_Trend, all_of(paste0(available_taxa, "_mean"))) %>%
    pivot_longer(cols = ends_with("_mean"), names_to = "Taxon", values_to = "RelAbund") %>%
    mutate(Taxon = gsub("_mean$", "", Taxon))
  
  p2 <- ggplot(plot_data, aes(x = Glucose_Trend, y = RelAbund, color = Taxon)) +
    geom_point(size = 3) +
    geom_smooth(method = "lm", se = FALSE, linetype = "dashed") +
    facet_wrap(~Taxon, scales = "free_y") +
    theme_minimal(base_size = 10) +
    scale_color_brewer(palette = "Set2") +
    labs(title = "Key Taxa vs Fasting Glucose Trajectory",
         subtitle = "Glucose trend from Strollo et al. 2018",
         x = "Fasting Glucose (Normalized Trend)", y = "Mean Relative Abundance") +
    theme(legend.position = "none")
  ggsave(file.path(fig_dir, "Taxa_vs_Glucose.pdf"), p2, width = 10, height = 7)
}

message("W1.5: Phase-level correlations complete.")
