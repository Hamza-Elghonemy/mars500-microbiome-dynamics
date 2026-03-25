# ==============================================================================
# M9: Comprehensive Visualizations (R)
# MARS500 Temporal Gut Microbiome Dynamics (Phase III)
# ==============================================================================
# Synthesizes all multi-omics pipelines into publication figures.
# - Taxonomy Heatmaps & Stacked Compositions
# - PICRUSt2 Functional PCoA & Pathway Heatmaps
# - Network Topological Disintegration Trajectory
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(ggplot2)
  library(pheatmap)
  library(tidyverse)
  library(vegan)
  library(igraph)
})

message("Starting Phase III: Multi-Omics Visual Synthesis...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
meta_dir <- file.path(base_dir, "data", "metadata")
fig_dir <- file.path(base_dir, "results", "figures")
net_dir <- file.path(base_dir, "results", "networks")
picrust_dir <- file.path(base_dir, "results", "picrust2", "pathways_out")

# 1. TAXONOMY HEATMAP & COMPOSITION ############################################
message("1. Generating Taxonomy Heatmaps...")

ps <- readRDS(file.path(data_dir, "phyloseq_obj.rds"))
meta <- read.table(file.path(meta_dir, "processed_metadata.tsv"), sep="\t", header=TRUE, row.names=1)
sample_data(ps) <- sample_data(meta)

# Remove NA Genera and agglomerate
ps_gen <- tax_glom(ps, taxrank="genus", NArm=TRUE)
taxa_names(ps_gen) <- tax_table(ps_gen)[,"genus"]

# Transform to relative abundance
ps_rel <- transform_sample_counts(ps_gen, function(x) x / sum(x))

# Top 25 most variable Genera
gen_vars <- apply(otu_table(ps_rel), 1, var)
top_gen <- names(sort(gen_vars, decreasing=TRUE))[1:30]
ps_top <- prune_taxa(top_gen, ps_rel)

# Prepare matrix for pheatmap
mat <- as(otu_table(ps_top), "matrix")
# Log transform (pseudo-count 1e-5)
mat_log <- log10(mat + 1e-5)

# Annotations (Phase)
anno_df <- data.frame(Phase = sample_data(ps_top)$Phase)
rownames(anno_df) <- sample_names(ps_top)

# Ensure phases are ordered logically
anno_df$Phase <- factor(anno_df$Phase, levels=c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission"))

# Custom Colors
phase_cols <- c("Pre-Mission"="#440154FF", "Early"="#3B528BFF", "Mid"="#21908CFF", "Late"="#5DC863FF", "Post-Mission"="#FDE725FF")
anno_colors <- list(Phase = phase_cols)

# Plot heatmap
pdf(file.path(fig_dir, "Taxonomy_Heatmap.pdf"), width = 12, height = 8)
pheatmap(mat_log, 
         annotation_col = anno_df, 
         annotation_colors = anno_colors,
         cluster_cols = FALSE, # Keep timeline intact instead of clustering samples
         main = "Top 30 Variable Genera Across MARS500 Timeline (Log10 Rel. Abund)",
         color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
         show_colnames = FALSE,
         fontsize_row = 8)
dev.off()


# 2. PICRUST2 FUNCTIONAL PCoA & HEATMAP ########################################
message("2. Generating Functional Pathway Visualizations...")

if(file.path(picrust_dir, "path_abun_unstrat.tsv.gz") %>% file.exists()) {
  path_raw <- read.table(gzfile(file.path(picrust_dir, "path_abun_unstrat.tsv.gz")), 
                         header=TRUE, sep="\t", row.names=1, check.names=FALSE)
  
  # Align with meta
  common <- intersect(colnames(path_raw), rownames(meta))
  path_mat <- path_raw[, common]
  meta_func <- meta[common, ]
  
  # Relative abundance & Transpose
  path_rel <- sweep(path_mat, 2, colSums(path_mat), "/")
  path_t <- t(path_rel)
  
  # PCoA
  bc_dist <- vegdist(path_t, method="bray")
  pcoa <- cmdscale(bc_dist, k=2, eig=TRUE)
  
  pcoa_df <- data.frame(
    PCoA1 = pcoa$points[,1],
    PCoA2 = pcoa$points[,2],
    Phase = factor(meta_func$Phase, levels=c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission")),
    Crew = meta_func$CrewMember
  )
  
  p_func <- ggplot(pcoa_df, aes(x=PCoA1, y=PCoA2, color=Phase, shape=Crew)) +
    geom_point(size=4, alpha=0.8) +
    scale_color_manual(values=phase_cols) +
    theme_minimal() +
    labs(title="Functional PCoA (MetaCyc Pathways)",
         subtitle="Bray-Curtis Distance of Imputed Metagenomes",
         x=paste0("PC1 (", round(100*pcoa$eig[1]/sum(pcoa$eig),1), "%)"),
         y=paste0("PC2 (", round(100*pcoa$eig[2]/sum(pcoa$eig),1), "%)"))
  
  ggsave(file.path(fig_dir, "Functional_PCoA.pdf"), p_func, width=8, height=6)
  
  # Functional Heatmap (Top 25 Pathways)
  path_vars <- apply(path_rel, 1, var)
  top_paths <- names(sort(path_vars, decreasing=TRUE))[1:25]
  
  mat_path <- as.matrix(path_rel[top_paths, ])
  mat_path_log <- log10(mat_path + 1e-6)
  
  anno_func <- data.frame(Phase = factor(meta_func$Phase, levels=c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission")))
  rownames(anno_func) <- rownames(meta_func)
  
  pdf(file.path(fig_dir, "Functional_Pathway_Heatmap.pdf"), width = 12, height = 7)
  pheatmap(mat_path_log,
           annotation_col = anno_func,
           annotation_colors = anno_colors,
           cluster_cols = FALSE,
           main = "Top 25 Variable MetaCyc Pathways",
           color = colorRampPalette(c("navy", "white", "darkorange"))(100),
           show_colnames = FALSE,
           fontsize_row = 7)
  dev.off()
} else {
  message("[WARNING] PICRUSt2 pathway file not found. Skipping functional metrics.")
}


# 3. NETWORK METRICS TRAJECTORY ################################################
message("3. Generating Network Topologies Trajectory...")

phases <- c("Pre-Mission", "Early", "Mid", "Late", "Post-Mission")
net_metrics <- data.frame()

for(ph in phases) {
  net_file <- file.path(net_dir, paste0("spearman_network_", ph, ".rds"))
  if(file.exists(net_file)) {
    ig <- readRDS(net_file)
    if(vcount(ig) > 1 && ecount(ig) > 0) {
      if("weight" %in% edge_attr_names(ig)) {
        E(ig)$weight <- abs(E(ig)$weight)
      }
      
      den <- edge_density(ig)
      fc <- cluster_louvain(ig, weights=NA)
      mod <- modularity(fc)
      m_deg <- mean(degree(ig))
      
      net_metrics <- rbind(net_metrics, data.frame(
        Phase=ph, Density=den, Modularity=mod, Mean_Degree=m_deg
      ))
    }
  }
}

if(nrow(net_metrics) > 0) {
  net_metrics$Phase <- factor(net_metrics$Phase, levels=phases)
  net_metrics$TimePoint <- as.numeric(net_metrics$Phase)
  
  # Melt for facet plotting
  net_melt <- pivot_longer(net_metrics, cols=c("Density", "Modularity", "Mean_Degree"), names_to="Metric")
  
  p_net <- ggplot(net_melt, aes(x=Phase, y=value, group=Metric, color=Metric)) +
    geom_line(size=1.5) +
    geom_point(size=4) +
    facet_wrap(~Metric, scales="free_y", ncol=1) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle=45, hjust=1, size=12)) +
    scale_color_brewer(palette="Set1") +
    labs(title="Microbial Network Trajectory Disintegration",
         subtitle="Tracking the collapse of interactive topology during isolation.",
         x="", y="Value") +
    theme(legend.position="none")
    
  ggsave(file.path(fig_dir, "Network_Trajectory_Decay.pdf"), p_net, width=7, height=10)
}

message("Master Phase III Visualization pipeline completed successfully!")
