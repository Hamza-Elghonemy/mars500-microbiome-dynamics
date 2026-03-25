# ==============================================================================
# M6: Differential Abundance - Publication Standard (R)
# MARS500 Temporal Gut Microbiome Dynamics
# ==============================================================================
# Conducts rigorous pairwise DA testing using DESeq2 adapted for longitudinal 
# zero-inflated microbiome datasets. Generates Volcano Plots mapping LFC.
# ==============================================================================

suppressPackageStartupMessages({
  library(phyloseq)
  library(DESeq2)
  library(tidyverse)
  library(ggplot2)
  library(ggrepel)
})

message("Starting Advanced Differential Abundance Pipeline...")

base_dir <- getwd()
if(basename(base_dir) == "pipelines") base_dir <- dirname(base_dir)

data_dir <- file.path(base_dir, "data", "processed")
results_dir <- file.path(base_dir, "results")

ps_rds <- file.path(data_dir, "phyloseq_obj.rds")
if(file.exists(ps_rds)) {
  ps <- readRDS(ps_rds)
  
  # Agglomerate to Genus level for meaningful differential abundance
  ps_genus <- tax_glom(ps, taxrank = "genus", NArm = FALSE)
  
  # Filter heavily zero-inflated sparse taxa
  ps_genus <- filter_taxa(ps_genus, function(x) sum(x > 3) > (0.1 * length(x)), TRUE)
  
  message("Running DESeq2 Pairwise Testing (Pre-Mission vs. Late) ...")
  
  # Convert to DESeq2 format (Requires purely un-normalized count data natively)
  # Model formula: Control for CrewMember baseline differences, measure Phase effect
  dds <- phyloseq_to_deseq2(ps_genus, ~ CrewMember + Phase)
  
  # Calculate geometric means prior to estimate size factors (vital for microbiome sparsity)
  gm_mean = function(x, na.rm=TRUE){
    exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
  }
  geoMeans = apply(counts(dds), 1, gm_mean)
  dds <- estimateSizeFactors(dds, geoMeans = geoMeans)
  
  dds <- DESeq(dds, test="Wald", fitType="parametric")
  
  # Extract contrast (Late Confinement vs Baseline Pre-Mission)
  res <- results(dds, contrast=c("Phase", "Late", "Pre-Mission"))
  res_df <- as.data.frame(res)
  
  # Merge with taxonomy tables
  taxa_map <- as.data.frame(tax_table(ps_genus))
  res_df$Genus <- taxa_map[rownames(res_df), "genus"]
  res_df <- res_df %>% drop_na(padj) %>% arrange(padj)
  
  write.csv(res_df, file.path(results_dir, "tables", "DESeq2_Late_vs_PreMission.csv"))
  
  message("Generating Volcano Plot...")
  
  # Categorize significance (padj < 0.05 & absolute LFC > 1.5)
  res_df <- res_df %>% mutate(
    Category = case_when(
        padj < 0.05 & log2FoldChange > 1.5 ~ "Up-Regulated in Late",
        padj < 0.05 & log2FoldChange < -1.5 ~ "Down-Regulated in Late",
        TRUE ~ "Not Significant"
    )
  )
  
  # Plot
  p_volcano <- ggplot(res_df, aes(x=log2FoldChange, y=-log10(padj), color=Category)) +
    geom_point(alpha=0.8, size=2) +
    scale_color_manual(values=c("Up-Regulated in Late"="#D55E00", "Down-Regulated in Late"="#0072B2", "Not Significant"="grey")) +
    geom_vline(xintercept=c(-1.5, 1.5), linetype="dashed", color="black", alpha=0.5) +
    geom_hline(yintercept=-log10(0.05), linetype="dashed", color="black", alpha=0.5) +
    geom_text_repel(data=subset(res_df, Category != "Not Significant"), aes(label=Genus), size=3, max.overlaps = 15) +
    theme_minimal() +
    labs(title="Volcano Plot: Late Confinement vs. Pre-Mission", x="Log2 Fold Change", y="-Log10(adj P-Value)")
    
  ggsave(file.path(results_dir, "figures", "DA_Volcano_Late_vs_PreMission.pdf"), p_volcano, width = 10, height = 7)
  message("Differential Abundance testing complete. Exported Volcano Plot.")

} else {
  message("[WARNING] phyloseq_obj.rds not found! Please ensure pipeline 02 produced it.")
}
