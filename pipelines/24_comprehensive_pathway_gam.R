# ==============================================================================
# Comprehensive Pathway GAM Analysis (All Pathways)
# MARS500 Publication Pipeline
# ==============================================================================

suppressPackageStartupMessages({
  library(mgcv)
  library(tidyverse)
  library(ggplot2)
})

message("Starting Comprehensive Pathway GAM Analysis...")

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
message("Loading metadata and pathways...")
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"),
                   sep = "\t", header = TRUE, row.names = 1)

path_file <- file.path(picrust_dir, "path_abun_unstrat.tsv.gz")
if (!file.exists(path_file)) stop("Pathway file not found: ", path_file)

pathways <- t(read.table(gzfile(path_file), header = TRUE, sep = "\t", row.names = 1, check.names = FALSE))

common <- intersect(rownames(pathways), rownames(meta))
pathways <- pathways[common, ]
meta <- meta[common, ]

# Relative abundance
pathways_rel <- as.data.frame(sweep(pathways, 1, rowSums(pathways), "/"))

# ---------- Fit GAM for every pathway ----------------------------------------
message("Fitting GAMs for all ", ncol(pathways_rel), " pathways. This might take a minute...")

res_list <- list()
# Loop over all pathways
for(pw in colnames(pathways_rel)) {
  df <- data.frame(y = pathways_rel[[pw]], Day = meta$Timepoint_Day, CrewMember = as.factor(meta$CrewMember))
  df <- drop_na(df)
  
  if(var(df$y) == 0) next
  
  # Try Tweedie first for zero-inflated continuous data, fallback to Gaussian
  fit <- tryCatch({
    gam(y ~ s(Day, k = 8) + s(CrewMember, bs = "re"), data = df, method = "REML", family = tw())
  }, error = function(e) {
    tryCatch({
      gam(y ~ s(Day, k = 8) + s(CrewMember, bs = "re"), data = df, method = "REML")
    }, error = function(e) NULL)
  })
  
  if(!is.null(fit)) {
    sm <- summary(fit)
    res_list[[pw]] <- data.frame(
      Pathway = pw,
      EDF = sm$s.table[1, "edf"],
      F_stat = sm$s.table[1, "F"],
      p_value = sm$s.table[1, "p-value"],
      Dev_explained = sm$dev.expl * 100
    )
  }
}

res_df <- bind_rows(res_list)
res_df$p_adj <- p.adjust(res_df$p_value, method = "fdr")
res_df <- res_df %>% arrange(p_adj)

write.csv(res_df, file.path(tab_dir, "all_pathways_gam_results.csv"), row.names = FALSE)
message("Saved GAM results to all_pathways_gam_results.csv")

# ---------- Plot Top 12 Significant Pathways ---------------------------------
top_pw <- head(res_df$Pathway[res_df$p_adj < 0.05], 12)
if(length(top_pw) == 0) {
  message("No significant pathways found!")
  quit(save="no")
}

message("Plotting top ", length(top_pw), " significant pathways...")

path_pred <- data.frame()
for(pw in top_pw) {
  df <- data.frame(y = pathways_rel[[pw]], Day = meta$Timepoint_Day, CrewMember = as.factor(meta$CrewMember))
  fit <- gam(y ~ s(Day, k = 8) + s(CrewMember, bs = "re"), data = df, method = "REML", family = tw())
  
  nd <- expand.grid(Day = seq(min(df$Day), max(df$Day), length.out = 100), CrewMember = levels(df$CrewMember)[1])
  nd$fit <- predict(fit, newdata = nd, exclude = "s(CrewMember)", type = "response")
  nd$Pathway <- pw
  path_pred <- rbind(path_pred, nd)
}

plot_data <- pathways_rel %>% 
  select(all_of(top_pw)) %>% 
  rownames_to_column("Sample") %>%
  left_join(meta %>% rownames_to_column("Sample") %>% select(Sample, Timepoint_Day, CrewMember), by = "Sample") %>%
  pivot_longer(cols = all_of(top_pw), names_to = "Pathway", values_to = "RelAbund") %>%
  mutate(Pathway = factor(Pathway, levels = top_pw))

path_pred$Pathway <- factor(path_pred$Pathway, levels = top_pw)

p <- ggplot() +
  geom_point(data = plot_data, aes(x = Timepoint_Day, y = RelAbund, color = as.factor(CrewMember)), alpha = 0.4) +
  geom_line(data = path_pred, aes(x = Day, y = fit), color = "#D7191C", linewidth = 1.2) +
  facet_wrap(~Pathway, scales = "free_y", ncol = 4) + 
  scale_color_brewer(palette = "Dark2", name="CrewMember") +
  theme_minimal(base_size = 10) +
  labs(title = "Top Shifting Functional Pathways Across Isolation (FDR < 0.05)",
       x = "Day of Isolation", y = "Relative Abundance") +
  theme(legend.position = "bottom", strip.text = element_text(size=8, face="bold"))

ggsave(file.path(fig_dir, "Top_Shifting_Pathways_GAM.pdf"), p, width = 15, height = 10)
message("Saved top pathways plot.")
message("Done.")
