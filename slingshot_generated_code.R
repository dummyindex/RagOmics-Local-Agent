run <- function(path_dict, params) {
    library(Seurat)
    library(slingshot)
    library(SingleCellExperiment)
    library(tibble)
    
    # Construct file paths
    input_file <- file.path(path_dict$input_dir, "_node_seuratObject.rds")
    output_file <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    csv_output_file <- file.path(path_dict$output_dir, "pseudotime_results.csv")
    
    # Read input
    if (file.exists(input_file)) {
        seurat_obj <- readRDS(input_file)
    } else {
        stop(paste("Input file not found:", input_file))
    }
    
    # Check if PCA has been run
    if (!"pca" %in% Reductions(seurat_obj)) {
        stop("PCA has not been computed. Please run PCA before Slingshot.")
    }
    
    # Convert Seurat object to SingleCellExperiment
    sce <- as.SingleCellExperiment(seurat_obj)
    
    # Run Slingshot
    sce <- slingshot(sce, clusterLabels = 'seurat_clusters', reducedDim = 'PCA')
    
    # Extract pseudotime
    pseudotime <- slingPseudotime(sce)
    
    # Add pseudotime to Seurat object metadata
    seurat_obj[['pseudotime']] <- CreateAssayObject(counts = pseudotime)
    
    # Save pseudotime results to CSV
    pseudotime_df <- as.data.frame(pseudotime)
    pseudotime_df <- rownames_to_column(pseudotime_df, var = "Cell")
    write.csv(pseudotime_df, csv_output_file, row.names = FALSE)
    
    # Save updated Seurat object
    saveRDS(seurat_obj, output_file)
    cat("Output saved to:", output_file, "and pseudotime results to:", csv_output_file, "\n")
}