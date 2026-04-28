# ==============================================================================
# Functional GAM, Functional-Taxonomic Correlation, & Functional Diversity
# MARS500 Publication Pipeline
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(mgcv)
  library(tidyverse)
  library(ggplot2)
  library(vegan)
})

message("Starting Functional Analysis...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
meta_dir <- file.path(base_dir, "data", "metadata")
picrust_dir <- file.path(base_dir, "results", "picrust2", "pathways_out")
fig_dir  <- file.path(base_dir, "results", "figures")
tab_dir  <- file.path(base_dir, "results", "tables")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

# ---------- Load Data --------------------------------------------------------
message("Loading data...")
# Load phyloseq and metadata
ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"),
                   sep = "\t", header = TRUE, row.names = 1)
sample_data(ps) <- sample_data(meta)

# Load PiCRUST2 pathway abundances
path_file <- file.path(picrust_dir, "path_abun_unstrat.tsv.gz")
if (!file.exists(path_file)) stop("Pathway file not found: ", path_file)

pathways <- read.table(gzfile(path_file), header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
# pathways columns are samples, rows are pathways. Let's transpose.
pathways <- t(pathways)

# Ensure sample names match between metadata and pathways
common_samples <- intersect(rownames(pathways), rownames(meta))
pathways <- pathways[common_samples, ]
meta_sub <- meta[common_samples, ]

# Normalize pathways to relative abundance for fair comparison
pathways_rel <- sweep(pathways, 1, rowSums(pathways), "/")
pathways_rel_df <- as.data.frame(pathways_rel)

# ---------- Task 3: Functional vs Taxonomic Diversity -----------------------
message("Computing Functional Diversity...")
# Compute taxonomic Shannon diversity
tax_shannon <- estimate_richness(ps, measures = "Shannon")
tax_shannon <- tax_shannon[common_samples, , drop=FALSE]

# Compute functional Shannon diversity
func_shannon <- diversity(pathways, index = "shannon", MARGIN = 1)

div_df <- data.frame(
  SampleID = common_samples,
  Day = meta_sub$Timepoint_Day,
  CrewMember = as.factor(meta_sub$CrewMember),
  Phase = meta_sub$Phase,
  Taxonomic_Shannon = tax_shannon$Shannon,
  Functional_Shannon = func_shannon[common_samples]
)

# Plot diversity over time
div_long <- div_df %>%
  pivot_longer(cols = c("Taxonomic_Shannon", "Functional_Shannon"),
               names_to = "Diversity_Type", values_to = "Shannon_Index") %>%
  mutate(Diversity_Type = recode(Diversity_Type, 
                                 Taxonomic_Shannon = "Taxonomic Diversity",
                                 Functional_Shannon = "Functional Diversity"))

# Create prediction grid for GAM smoothing lines on the plot
div_preds <- data.frame()
for (div_type in unique(div_long$Diversity_Type)) {
  sub_df <- div_long[div_long$Diversity_Type == div_type, ]
  fit <- gam(Shannon_Index ~ s(Day, k = 10) + s(CrewMember, bs = "re"),
             data = sub_df, method = "REML")
  
  nd <- expand.grid(
    Day = seq(min(sub_df$Day), max(sub_df$Day), length.out = 200),
    CrewMember = levels(sub_df$CrewMember)[1]
  )
  nd$fit <- predict(fit, newdata = nd, exclude = "s(CrewMember)", type = "response")
  nd$Diversity_Type <- div_type
  div_preds <- rbind(div_preds, nd)
}

p_div <- ggplot() +
  geom_point(data = div_long, aes(x = Day, y = Shannon_Index, color = Diversity_Type),
             alpha = 0.4, size = 1.5) +
  geom_line(data = div_preds, aes(x = Day, y = fit, color = Diversity_Type),
            linewidth = 1.2) +
  scale_color_manual(values = c("Taxonomic Diversity" = "#D7191C", "Functional Diversity" = "#2C7BB6")) +
  theme_minimal(base_size = 12) +
  labs(title = "Taxonomic vs Functional Diversity Across Isolation",
       x = "Day of Isolation", y = "Shannon Diversity Index") +
  theme(legend.position = "bottom", legend.title = element_blank())

ggsave(file.path(fig_dir, "Taxonomic_vs_Functional_Diversity.pdf"), p_div, width = 8, height = 5)
message("  Saved Taxonomic vs Functional Diversity plot.")

# ---------- Task 2: Taxonomic-Functional Correlation ------------------------
message("Computing Taxonomic-Functional Correlation...")

# Get relative abundances of key taxa (Faecalibacterium, Sutterella)
rn <- rank_names(ps)
gr <- rn[grep("genus", rn, ignore.case = TRUE)]
if(length(gr) == 0) gr <- rn[length(rn)]
ps_gen <- tax_glom(ps, taxrank = gr, NArm = TRUE)
taxa_names(ps_gen) <- tax_table(ps_gen)[, gr]
ps_gen_rel <- transform_sample_counts(ps_gen, function(x) x / sum(x))
gen_mat <- as.data.frame(t(otu_table(ps_gen_rel)))[common_samples, ]

cor_taxa <- c("Faecalibacterium", "Sutterella")
cor_pathways <- c("FERMENTATION-PWY", "LPSSYN-PWY", "TRPSYN-PWY")

cor_results <- data.frame()
for (taxon in cor_taxa) {
  if (taxon %in% colnames(gen_mat)) {
    for (path in cor_pathways) {
      if (path %in% colnames(pathways_rel_df)) {
        res <- cor.test(gen_mat[[taxon]], pathways_rel_df[[path]], method = "pearson")
        cor_results <- rbind(cor_results, data.frame(
          Taxon = taxon, Pathway = path, r = res$estimate, p_value = res$p.value
        ))
      }
    }
  }
}

write.csv(cor_results, file.path(tab_dir, "taxonomic_functional_correlations.csv"), row.names = FALSE)
message("  Saved correlations to taxonomic_functional_correlations.csv")
print(cor_results)

# ---------- Task 1: Functional GAM ------------------------------------------
message("Running Functional GAM on Key Pathways...")

# Find vitamin synthesis pathways
vitamin_pathways <- grep("VITAMIN|B12|FOLATE|BIOTIN|RIBOFLAVIN|THIAMIN", colnames(pathways_rel_df), ignore.case = TRUE, value = TRUE)
message("Found vitamin pathways: ", paste(vitamin_pathways, collapse=", "))

target_pathways <- unique(c(cor_pathways, vitamin_pathways))
available_pathways <- target_pathways[target_pathways %in% colnames(pathways_rel_df)]

# If there are too many vitamin pathways, just select a few representative ones to avoid massive plots
if (length(available_pathways) > 10) {
  available_pathways <- c(cor_pathways, available_pathways[4:8]) # Pick first few
}

gam_path_results <- data.frame()
path_pred_all <- data.frame()

for (path in available_pathways) {
  df <- data.frame(
    y = pathways_rel_df[[path]],
    Day = meta_sub$Timepoint_Day,
    CrewMember = as.factor(meta_sub$CrewMember)
  )
  df <- df[!is.na(df$y) & !is.na(df$Day), ]
  
  fit <- tryCatch({
    gam(y ~ s(Day, k = 8) + s(CrewMember, bs = "re"),
        data = df, method = "REML", family = tw()) # Tweedie for bounded/zero-inflated
  }, error = function(e) {
    gam(y ~ s(Day, k = 8) + s(CrewMember, bs = "re"),
        data = df, method = "REML")
  })
  
  sm <- summary(fit)
  gam_path_results <- rbind(gam_path_results, data.frame(
    Pathway = path,
    EDF = sm$s.table[1, "edf"],
    F_stat = sm$s.table[1, "F"],
    p_value = sm$s.table[1, "p-value"],
    Dev_explained = round(sm$dev.expl * 100, 1)
  ))
  
  nd <- expand.grid(
    Day = seq(min(df$Day), max(df$Day), length.out = 200),
    CrewMember = levels(df$CrewMember)[1]
  )
  nd$fit <- predict(fit, newdata = nd, exclude = "s(CrewMember)", type = "response")
  nd$Pathway <- path
  path_pred_all <- rbind(path_pred_all, nd)
}

write.csv(gam_path_results, file.path(tab_dir, "gam_pathways_results.csv"), row.names = FALSE)

# Plot Pathway Trajectories
path_obs <- data.frame(Day = meta_sub$Timepoint_Day, CrewMember = as.factor(meta_sub$CrewMember))
path_obs <- cbind(path_obs, pathways_rel_df[, available_pathways, drop=FALSE])

path_long <- path_obs %>%
  pivot_longer(cols = all_of(available_pathways), names_to = "Pathway", values_to = "RelAbund")

p_path <- ggplot() +
  geom_point(data = path_long, aes(x = Day, y = RelAbund, color = CrewMember),
             alpha = 0.4, size = 1.5) +
  geom_line(data = path_pred_all, aes(x = Day, y = fit),
            color = "#D7191C", linewidth = 1) +
  facet_wrap(~Pathway, scales = "free_y", ncol = 3) +
  scale_color_brewer(palette = "Dark2") +
  theme_minimal(base_size = 10) +
  labs(title = "GAM-fitted Longitudinal Trajectories of Key Functional Pathways",
       x = "Day of Isolation", y = "Relative Abundance") +
  theme(legend.position = "bottom")

ggsave(file.path(fig_dir, "Longitudinal_GAM_Pathways.pdf"), p_path, width = 12, height = 8)
message("  Saved functional GAM figure.")

message("Done.")
