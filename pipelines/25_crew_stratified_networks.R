# ==============================================================================
# Crew-Stratified Network Analysis
# MARS500 Publication Pipeline
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(igraph)
  library(tidyverse)
  library(ggplot2)
})

message("Starting Crew-Stratified Network Analysis...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
meta_dir <- file.path(base_dir, "data", "metadata")
fig_dir  <- file.path(base_dir, "results", "figures")
tab_dir  <- file.path(base_dir, "results", "tables")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

# ---------- Load Data --------------------------------------------------------
message("Loading phyloseq object...")
ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"),
                   sep = "\t", header = TRUE, row.names = 1)
sample_data(ps) <- sample_data(meta)

# Collapse to Genus level
rn <- rank_names(ps)
gr <- rn[grep("genus", rn, ignore.case = TRUE)]
if(length(gr) == 0) gr <- rn[length(rn)]
ps_gen <- tax_glom(ps, taxrank = gr, NArm = TRUE)
taxa_names(ps_gen) <- tax_table(ps_gen)[, gr]

crews <- sort(unique(meta$CrewMember))
net_metrics <- data.frame()
plot_list <- list()

message("Building networks per crew member...")
for(crew in crews) {
  # Subset to crew member
  ps_sub <- subset_samples(ps_gen, CrewMember == crew)
  
  # Prevalence filtering (present in >20% of crew's samples)
  ps_sub <- filter_taxa(ps_sub, function(x) sum(x > 0) > (0.2 * nsamples(ps_sub)), TRUE)
  mat <- as.data.frame(t(otu_table(ps_sub)))
  
  if(ncol(mat) < 10) {
    message("Crew ", crew, " has too few taxa. Skipping.")
    next
  }
  
  # Relative abundance transformation
  mat_rel <- sweep(mat, 1, rowSums(mat), "/")
  
  # Spearman correlation (for speed and simplicity in R, robust for monotonic relationships)
  # Keeping edges with |r| > 0.6
  cmat <- cor(mat_rel, method = "spearman", use = "pairwise.complete.obs")
  cmat[is.na(cmat)] <- 0
  cmat[abs(cmat) < 0.6] <- 0
  diag(cmat) <- 0
  
  # Build graph
  g <- graph_from_adjacency_matrix(cmat, mode = "undirected", weighted = TRUE, diag = FALSE)
  
  # Remove isolated nodes
  g <- delete_vertices(g, degree(g) == 0)
  
  if(vcount(g) > 0) {
    # Extract metrics
    
    # Create graph with absolute weights for Louvain clustering
    g_pos <- g
    E(g_pos)$weight <- abs(E(g)$weight)
    
    net_metrics <- rbind(net_metrics, data.frame(
      CrewMember = crew,
      Nodes = vcount(g),
      Edges = ecount(g),
      Density = edge_density(g),
      Modularity = modularity(cluster_louvain(g_pos)),
      Transitivity = transitivity(g)
    ))
    
    # Save a basic network plot to PDF
    pdf(file.path(fig_dir, paste0("Network_Crew_", crew, ".pdf")), width = 8, height = 8)
    # Define edge colors (red for negative, blue for positive)
    E(g)$color <- ifelse(E(g)$weight > 0, rgb(0,0,1,0.5), rgb(1,0,0,0.5))
    plot(g, vertex.label = NA, vertex.size = 5, vertex.color = "gray80",
         edge.width = abs(E(g)$weight)*2, main = paste("Crew", crew, "Microbiome Network"))
    dev.off()
  } else {
    message("Crew ", crew, " yielded an empty network after thresholding.")
  }
}

# Save metrics
write.csv(net_metrics, file.path(tab_dir, "crew_stratified_network_metrics.csv"), row.names = FALSE)
message("Saved network metrics to crew_stratified_network_metrics.csv")

# Plot metrics comparison
net_long <- net_metrics %>%
  pivot_longer(cols = c("Density", "Modularity", "Transitivity"), 
               names_to = "Metric", values_to = "Value")

p_metrics <- ggplot(net_long, aes(x = as.factor(CrewMember), y = Value, fill = as.factor(CrewMember))) +
  geom_bar(stat = "identity", color="black", alpha=0.8) +
  facet_wrap(~Metric, scales = "free_y") +
  scale_fill_brewer(palette = "Set3", guide = "none") +
  theme_minimal(base_size = 12) +
  labs(title = "Network Topological Variation Across Crew Members",
       x = "Crew Member", y = "Metric Value")

ggsave(file.path(fig_dir, "Network_Metrics_by_Crew.pdf"), p_metrics, width = 10, height = 4)
message("Saved metrics variation plot.")
message("Done.")
