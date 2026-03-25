# ==============================================================================
# W3.2: Longitudinal GAM / Spline Modeling
# MARS500 Publication Pipeline
# ==============================================================================
# Fits Generalized Additive Models (GAMs) with thin-plate splines to continuous
# Timepoint_Day for: (1) Shannon diversity, (2) key DA taxa, and (3) network
# metrics. CrewMember is included as a random intercept to control for
# inter-individual variation.
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(mgcv)
  library(tidyverse)
  library(ggplot2)
})

message("W3.2: Longitudinal Spline / GAM Modeling...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
meta_dir <- file.path(base_dir, "data", "metadata")
net_dir  <- file.path(base_dir, "results", "networks")
tab_dir  <- file.path(base_dir, "results", "tables")
fig_dir  <- file.path(base_dir, "results", "figures")

# ---------- Load data --------------------------------------------------------
ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"),
                   sep = "\t", header = TRUE, row.names = 1)
meta$Phase <- factor(meta$Phase,
                     levels = c("Pre-Mission","Early","Mid","Late","Post-Mission"))
sample_data(ps) <- sample_data(meta)

# Alpha diversity
alpha <- estimate_richness(ps, measures = c("Shannon", "Simpson"))
alpha$SampleID   <- rownames(alpha)
alpha$Day         <- meta[alpha$SampleID, "Timepoint_Day"]
alpha$CrewMember  <- factor(meta[alpha$SampleID, "CrewMember"])
alpha$Phase       <- meta[alpha$SampleID, "Phase"]

# Genus-level relative abundance for key taxa
rn <- rank_names(ps)
gr <- rn[grep("genus", rn, ignore.case = TRUE)]
if(length(gr) == 0) gr <- rn[length(rn)]
ps_gen <- tax_glom(ps, taxrank = gr, NArm = TRUE)
taxa_names(ps_gen) <- tax_table(ps_gen)[, gr]
ps_gen_rel <- transform_sample_counts(ps_gen, function(x) x / sum(x))
gen_mat <- as.data.frame(t(otu_table(ps_gen_rel)))
gen_mat$Day <- meta[rownames(gen_mat), "Timepoint_Day"]
gen_mat$CrewMember <- factor(meta[rownames(gen_mat), "CrewMember"])

# =============================================================================
# PART 1: GAM on Shannon Diversity ~ s(Day) + s(CrewMember, bs="re")
# =============================================================================
message("  → GAM: Shannon ~ s(Day) + random(CrewMember)...")

gam_shannon <- gam(Shannon ~ s(Day, k = 10) + s(CrewMember, bs = "re"),
                   data = alpha, method = "REML")

summary_shan <- summary(gam_shannon)
message("  Shannon GAM edf = ", round(summary_shan$s.table[1, "edf"], 2),
        ", p = ", format(summary_shan$s.table[1, "p-value"], digits = 3))

# Save GAM summary
sink(file.path(tab_dir, "gam_shannon_summary.txt"))
cat("=== GAM: Shannon ~ s(Day) + s(CrewMember, bs='re') ===\n\n")
print(summary_shan)
cat("\nAIC:", AIC(gam_shannon), "\n")
cat("Deviance explained:", round(summary_shan$dev.expl * 100, 1), "%\n")
sink()

# Prediction grid
new_data <- expand.grid(
  Day = seq(min(alpha$Day), max(alpha$Day), length.out = 200),
  CrewMember = levels(factor(alpha$CrewMember))[1]  # representative crew
)
# Population-level prediction (exclude random effect)
new_data$fit <- predict(gam_shannon, newdata = new_data, exclude = "s(CrewMember)",
                        type = "response")

# Plot
phase_breaks <- data.frame(
  xmin = c(-10, 1, 60, 360, 520),
  xmax = c(0, 59, 359, 519, 700),
  Phase = c("Pre-Mission","Early","Mid","Late","Post-Mission")
)

p1 <- ggplot() +
  geom_rect(data = phase_breaks,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = Phase),
            alpha = 0.15) +
  scale_fill_brewer(palette = "Set2") +
  geom_point(data = alpha, aes(x = Day, y = Shannon, color = CrewMember),
             alpha = 0.6, size = 2) +
  geom_line(data = new_data, aes(x = Day, y = fit), color = "#D7191C",
            linewidth = 1.2) +
  scale_color_brewer(palette = "Dark2") +
  theme_minimal(base_size = 12) +
  labs(title = "Longitudinal GAM: Shannon Diversity Across 520-Day Isolation",
       subtitle = paste0("Smooth spline, EDF = ",
                         round(summary_shan$s.table[1, "edf"], 1),
                         ", Dev. explained = ",
                         round(summary_shan$dev.expl * 100, 1), "%"),
       x = "Day of Isolation", y = "Shannon Diversity Index") +
  theme(legend.position = "right")

ggsave(file.path(fig_dir, "Longitudinal_GAM_Shannon.pdf"), p1, width = 11, height = 6)
message("  Saved Shannon GAM figure.")

# =============================================================================
# PART 2: GAM on key taxa (top DA genera)
# =============================================================================
message("  → GAM: Key taxa trajectories...")

key_taxa <- c("Sutterella", "Bilophila", "Faecalibacterium",
              "Bacteroides", "Dialister", "Akkermansia")
available <- intersect(key_taxa, colnames(gen_mat))

gam_taxa_results <- data.frame()
taxa_pred_all    <- data.frame()

for(taxon in available) {
  df <- data.frame(
    y = gen_mat[[taxon]],
    Day = gen_mat$Day,
    CrewMember = gen_mat$CrewMember
  )
  df <- df[!is.na(df$y) & !is.na(df$Day), ]

  fit <- tryCatch({
    gam(y ~ s(Day, k = 8) + s(CrewMember, bs = "re"),
        data = df, method = "REML", family = tw())  # Tweedie for zero-inflated
  }, error = function(e) {
    gam(y ~ s(Day, k = 8) + s(CrewMember, bs = "re"),
        data = df, method = "REML")
  })

  sm <- summary(fit)
  gam_taxa_results <- rbind(gam_taxa_results, data.frame(
    Taxon = taxon,
    EDF = sm$s.table[1, "edf"],
    F_stat = sm$s.table[1, "F"],
    p_value = sm$s.table[1, "p-value"],
    Dev_explained = round(sm$dev.expl * 100, 1),
    AIC = AIC(fit)
  ))

  nd <- data.frame(Day = seq(min(df$Day), max(df$Day), length.out = 200),
                   CrewMember = levels(factor(df$CrewMember))[1])
  nd$fit <- predict(fit, newdata = nd, exclude = "s(CrewMember)", type = "response")
  nd$Taxon <- taxon
  taxa_pred_all <- rbind(taxa_pred_all, nd)
}

write.csv(gam_taxa_results, file.path(tab_dir, "gam_taxa_results.csv"), row.names = FALSE)

# Multi-panel taxa trajectory plot
taxa_obs <- gen_mat %>%
  select(Day, CrewMember, all_of(available)) %>%
  pivot_longer(cols = all_of(available), names_to = "Taxon", values_to = "RelAbund")

p2 <- ggplot() +
  geom_point(data = taxa_obs, aes(x = Day, y = RelAbund, color = CrewMember),
             alpha = 0.4, size = 1.5) +
  geom_line(data = taxa_pred_all, aes(x = Day, y = fit),
            color = "#D7191C", linewidth = 1) +
  facet_wrap(~Taxon, scales = "free_y", ncol = 2) +
  scale_color_brewer(palette = "Dark2") +
  theme_minimal(base_size = 10) +
  labs(title = "GAM-fitted Longitudinal Trajectories of Key Biomarker Genera",
       subtitle = "Thin-plate splines + crew random intercepts (Tweedie family)",
       x = "Day of Isolation", y = "Relative Abundance") +
  theme(legend.position = "bottom")

ggsave(file.path(fig_dir, "Longitudinal_GAM_Taxa.pdf"), p2, width = 10, height = 8)
message("  Saved taxa GAM figure.")

# =============================================================================
# PART 3: Simpson diversity GAM (for comparison)
# =============================================================================
message("  → GAM: Simpson diversity...")

gam_simpson <- gam(Simpson ~ s(Day, k = 10) + s(CrewMember, bs = "re"),
                   data = alpha, method = "REML")
sm_simp <- summary(gam_simpson)

nd_simp <- expand.grid(
  Day = seq(min(alpha$Day), max(alpha$Day), length.out = 200),
  CrewMember = levels(factor(alpha$CrewMember))[1]
)
nd_simp$fit <- predict(gam_simpson, newdata = nd_simp, exclude = "s(CrewMember)",
                       type = "response")

p3 <- ggplot() +
  geom_rect(data = phase_breaks,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = Phase),
            alpha = 0.15) +
  scale_fill_brewer(palette = "Set2") +
  geom_point(data = alpha, aes(x = Day, y = Simpson, color = CrewMember),
             alpha = 0.6, size = 2) +
  geom_line(data = nd_simp, aes(x = Day, y = fit), color = "#2C7BB6",
            linewidth = 1.2) +
  scale_color_brewer(palette = "Dark2") +
  theme_minimal(base_size = 12) +
  labs(title = "Longitudinal GAM: Simpson Evenness Across 520-Day Isolation",
       subtitle = paste0("EDF = ", round(sm_simp$s.table[1, "edf"], 1),
                         ", Dev. explained = ",
                         round(sm_simp$dev.expl * 100, 1), "%"),
       x = "Day of Isolation", y = "Simpson Evenness Index") +
  theme(legend.position = "right")

ggsave(file.path(fig_dir, "Longitudinal_GAM_Simpson.pdf"), p3, width = 11, height = 6)

# =============================================================================
# Save consolidated summary
# =============================================================================
sink(file.path(tab_dir, "gam_summary_all.txt"))
cat("=== MARS500 Longitudinal GAM Summary ===\n\n")
cat("Shannon GAM:\n")
cat("  EDF:", round(summary_shan$s.table[1, "edf"], 2), "\n")
cat("  p-value:", format(summary_shan$s.table[1, "p-value"], digits = 3), "\n")
cat("  Deviance explained:", round(summary_shan$dev.expl * 100, 1), "%\n")
cat("  AIC:", AIC(gam_shannon), "\n\n")
cat("Simpson GAM:\n")
cat("  EDF:", round(sm_simp$s.table[1, "edf"], 2), "\n")
cat("  p-value:", format(sm_simp$s.table[1, "p-value"], digits = 3), "\n")
cat("  Deviance explained:", round(sm_simp$dev.expl * 100, 1), "%\n")
cat("  AIC:", AIC(gam_simpson), "\n\n")
cat("Taxa GAMs:\n")
print(gam_taxa_results)
sink()

message("W3.2: Longitudinal GAM modeling complete.")
