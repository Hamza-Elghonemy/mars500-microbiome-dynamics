# ==============================================================================
# W3.3: Keystone Species Statistics
# MARS500 Publication Pipeline
# ==============================================================================
# Formal hub score calculation (degree, betweenness, closeness, eigenvector)
# with permutation test for significance and tracking keystone persistence.
# ==============================================================================

suppressPackageStartupMessages({
  library(igraph)
  library(tidyverse)
  library(ggplot2)
})

message("W3.3: Keystone Species Statistics...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

net_dir <- file.path(base_dir, "results", "networks")
tab_dir <- file.path(base_dir, "results", "tables")
fig_dir <- file.path(base_dir, "results", "figures")

phases <- c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission")

# ---------- 1. Calculate centrality metrics per phase -------------------------
all_centrality <- data.frame()

for(ph in phases) {
  net_file <- file.path(net_dir, paste0("spearman_network_", ph, ".rds"))
  if(!file.exists(net_file)) next
  
  ig <- readRDS(net_file)
  if(vcount(ig) == 0 || ecount(ig) == 0) next
  
  if("weight" %in% edge_attr_names(ig)) E(ig)$weight <- abs(E(ig)$weight)
  
  deg <- degree(ig)
  btw <- betweenness(ig, weights = NA)  # Unweighted for topological betweenness
  cls <- closeness(ig, weights = NA)
  eig <- tryCatch(eigen_centrality(ig, weights = NA)$vector, error = function(e) rep(NA, vcount(ig)))
  hub <- hub_score(ig, weights = NA)$vector
  
  phase_df <- data.frame(
    Phase = ph,
    Taxon = V(ig)$name,
    Degree = deg,
    Betweenness = btw,
    Closeness = cls,
    Eigenvector = eig,
    Hub_Score = hub,
    stringsAsFactors = FALSE
  )
  
  all_centrality <- rbind(all_centrality, phase_df)
}

write.csv(all_centrality, file.path(tab_dir, "centrality_all_phases.csv"), row.names = FALSE)
message("  Saved full centrality table.")

# ---------- 2. Identify keystones (95th percentile) per phase ----------------
keystones <- all_centrality %>%
  group_by(Phase) %>%
  filter(Degree >= quantile(Degree, 0.95, na.rm = TRUE) |
         Betweenness >= quantile(Betweenness, 0.95, na.rm = TRUE)) %>%
  arrange(Phase, desc(Betweenness)) %>%
  ungroup()

write.csv(keystones, file.path(tab_dir, "keystone_taxa_formal.csv"), row.names = FALSE)
message("  Saved formal keystone taxa.")

# ---------- 3. Permutation test: Are keystones more central than random? ------
message("  Running permutation tests (1000 iterations per phase)...")

perm_results <- data.frame()

for(ph in phases) {
  net_file <- file.path(net_dir, paste0("spearman_network_", ph, ".rds"))
  if(!file.exists(net_file)) next
  
  ig <- readRDS(net_file)
  if(vcount(ig) < 10 || ecount(ig) == 0) next
  if("weight" %in% edge_attr_names(ig)) E(ig)$weight <- abs(E(ig)$weight)
  
  obs_btw <- betweenness(ig, weights = NA)
  obs_mean_top5 <- mean(sort(obs_btw, decreasing = TRUE)[1:5])
  
  n_perm <- 1000
  null_means <- numeric(n_perm)
  
  for(k in 1:n_perm) {
    # Randomly rewire edges while preserving degree distribution
    ig_rand <- rewire(ig, with = keeping_degseq(niter = ecount(ig) * 10))
    rand_btw <- betweenness(ig_rand, weights = NA)
    null_means[k] <- mean(sort(rand_btw, decreasing = TRUE)[1:5])
  }
  
  p_val <- sum(null_means >= obs_mean_top5) / n_perm
  
  perm_results <- rbind(perm_results, data.frame(
    Phase = ph,
    Observed_Top5_Betweenness = obs_mean_top5,
    Null_Mean = mean(null_means),
    Null_SD = sd(null_means),
    Z_score = (obs_mean_top5 - mean(null_means)) / sd(null_means),
    P_value = p_val
  ))
}

write.csv(perm_results, file.path(tab_dir, "keystone_permutation_test.csv"), row.names = FALSE)
message("  Saved keystone permutation test results.")

# ---------- 4. Keystone persistence across phases ----------------------------
# Track which taxa appear as keystones in multiple phases
persistence <- keystones %>%
  group_by(Taxon) %>%
  summarize(
    N_Phases_Keystone = n_distinct(Phase),
    Phases = paste(Phase, collapse = ", "),
    Mean_Betweenness = mean(Betweenness, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(N_Phases_Keystone), desc(Mean_Betweenness))

write.csv(persistence, file.path(tab_dir, "keystone_persistence.csv"), row.names = FALSE)

# ---------- 5. Visualization: Keystone stability plot -------------------------
if(nrow(persistence) > 0) {
  top_persistent <- persistence %>% slice_max(N_Phases_Keystone, n = 15)
  
  # Create a presence/absence matrix
  ks_wide <- keystones %>%
    filter(Taxon %in% top_persistent$Taxon) %>%
    select(Phase, Taxon, Betweenness) %>%
    pivot_wider(names_from = Phase, values_from = Betweenness, values_fill = 0)
  
  # Melt for plotting
  ks_long <- keystones %>%
    filter(Taxon %in% top_persistent$Taxon) %>%
    mutate(Phase = factor(Phase, levels = phases))
  
  p <- ggplot(ks_long, aes(x = Phase, y = reorder(Taxon, Betweenness), fill = Betweenness)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "#FEE0D2", high = "#DE2D26", na.value = "grey90") +
    theme_minimal(base_size = 10) +
    labs(title = "Keystone Taxa Stability Across Mission Phases",
         subtitle = "Betweenness centrality of top hub taxa (95th percentile)",
         x = "", y = "", fill = "Betweenness\nCentrality") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(file.path(fig_dir, "Keystone_Stability.pdf"), p, width = 9, height = 7)
  message("  Saved keystone stability heatmap.")
}

message("W3.3: Keystone species statistics complete.")
