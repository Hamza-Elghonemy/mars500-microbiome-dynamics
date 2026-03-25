# ==============================================================================
# M7: Network Visualization & Keystone Taxa extraction (R)
# MARS500 Temporal Gut Microbiome Dynamics
# ==============================================================================
# Calculates topological benchmarks (Degree, Betweenness) against the 5 arrays.
# Plots high-quality publication diagrams utilizing base `igraph` and highlights hubs.
# ==============================================================================

suppressPackageStartupMessages({
  library(igraph)
})

message("Starting Network Topology & Visualization Pipeline...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

networks_dir <- file.path(base_dir, "results", "networks")
fig_dir <- file.path(base_dir, "results", "figures")
tab_dir <- file.path(base_dir, "results", "tables")

rds_files <- list.files(networks_dir, pattern="spearman_network_.*\\.rds$", full.names=TRUE)

if(length(rds_files) > 0) {
  
  all_keystone <- data.frame()
  
  for(rds in rds_files) {
    phase <- gsub("spearman_network_", "", basename(rds))
    phase <- gsub("\\.rds", "", phase)
    
    cat("Analyzing Topology for Phase:", phase, "\n")
    
    ig <- readRDS(rds)
    
    if(vcount(ig) > 0 && ecount(ig) > 0) {
      
      # Convert all edge weights to absolute values so topology layouts/algorithms don't crash
      if("weight" %in% edge_attr_names(ig)) {
          E(ig)$weight <- abs(E(ig)$weight)
      }
      
      V(ig)$Degree <- degree(ig)
      V(ig)$Betweenness <- betweenness(ig, weights=NA)
      
      fc <- cluster_louvain(ig, weights=NA)
      V(ig)$Community <- membership(fc)
      
      ig <- delete_vertices(ig, V(ig)[Degree == 0])
      
      if(vcount(ig) > 0) {
        deg_thresh <- quantile(V(ig)$Degree, 0.95)
        V(ig)$Is_Keystone <- ifelse(V(ig)$Degree >= deg_thresh, "Keystone", "Standard")
        
        # Build DataFrame
        df <- data.frame(
          name = V(ig)$name,
          Degree = V(ig)$Degree,
          Betweenness = V(ig)$Betweenness,
          Community = V(ig)$Community,
          Is_Keystone = V(ig)$Is_Keystone,
          Phase = phase
        )
        keystones <- subset(df, Is_Keystone == "Keystone")
        all_keystone <- rbind(all_keystone, keystones)
        
        # Plot Network
        V(ig)$color <- ifelse(V(ig)$Is_Keystone == "Keystone", "red", "lightblue")
        V(ig)$size <- log(V(ig)$Degree + 1) * 3
        V(ig)$label <- ifelse(V(ig)$Is_Keystone == "Keystone", V(ig)$name, NA)
        
        pdf(file.path(fig_dir, paste0("Network_", phase, "_Visualization.pdf")), width = 10, height = 8)
        plot(ig, layout=layout_with_fr, main=paste("Co-occurrence Network:", phase),
             vertex.label.cex=0.6, edge.color="grey80", vertex.label.color="black")
        dev.off()
        
      } else {
        message("  -> Network is completely sparse (no edges passing threshold).")
      }
    } else {
       message("  -> Graph object empty.")
    }
  }
  
  write.csv(all_keystone, file.path(tab_dir, "Keystone_Taxa_Across_Phases.csv"), row.names=FALSE)
  message("Extracted Topological networks and saved Keystone summaries.")
  
} else {
  message("[WARNING] No network .rds files found. Ensure pipeline 04 generated them.")
}
