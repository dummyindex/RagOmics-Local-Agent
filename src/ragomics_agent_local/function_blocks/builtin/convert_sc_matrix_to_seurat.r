run <- function(path_dict, params) {
    # Load required libraries
    library(Seurat)
    library(Matrix)
    library(jsonlite)
    
    # Read SC matrix format
    sc_dir <- file.path(path_dict$input_dir, "_node_sc_matrix")
    
    if (!dir.exists(sc_dir)) {
        stop("No _node_sc_matrix found in input")
    }
    
    # Read metadata
    metadata <- fromJSON(file.path(sc_dir, "metadata.json"))
    
    # Read cell and gene names
    obs_names <- readLines(file.path(sc_dir, "obs_names.txt"))
    var_names <- readLines(file.path(sc_dir, "var_names.txt"))
    
    # Read expression matrix
    if (file.exists(file.path(sc_dir, "X.mtx"))) {
        X <- readMM(file.path(sc_dir, "X.mtx"))
    } else if (file.exists(file.path(sc_dir, "X.csv"))) {
        X <- as.matrix(read.csv(file.path(sc_dir, "X.csv"), row.names = 1))
    } else {
        stop("No expression matrix found (X.mtx or X.csv)")
    }
    
    # Note: X is already in cells x genes format from Python
    # But Seurat expects genes x cells, so transpose
    X <- t(X)
    rownames(X) <- var_names
    colnames(X) <- obs_names
    
    # Create Seurat object
    seurat_obj <- CreateSeuratObject(counts = X, project = "converted_from_sc_matrix")
    
    # Add cell metadata if available
    obs_dir <- file.path(sc_dir, "obs")
    if (dir.exists(obs_dir)) {
        obs_files <- list.files(obs_dir, pattern = "\\.csv$", full.names = TRUE)
        
        for (obs_file in obs_files) {
            col_name <- tools::file_path_sans_ext(basename(obs_file))
            obs_data <- read.csv(obs_file)
            
            if (nrow(obs_data) == length(obs_names)) {
                # Add to Seurat metadata
                seurat_obj[[col_name]] <- obs_data[, 1]
            }
        }
    }
    
    # Add gene metadata if available
    var_dir <- file.path(sc_dir, "var")
    if (dir.exists(var_dir)) {
        var_files <- list.files(var_dir, pattern = "\\.csv$", full.names = TRUE)
        
        for (var_file in var_files) {
            col_name <- tools::file_path_sans_ext(basename(var_file))
            var_data <- read.csv(var_file)
            
            if (nrow(var_data) == length(var_names)) {
                # Add to Seurat feature metadata
                seurat_obj[["RNA"]][[col_name]] <- var_data[, 1]
            }
        }
    }
    
    # Save Seurat object
    output_file <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    saveRDS(seurat_obj, output_file)
    
    # Also preserve the original SC matrix
    file.copy(sc_dir, path_dict$output_dir, recursive = TRUE)
    
    # Print summary
    cat(sprintf("Created Seurat object with %d cells and %d genes\n", 
                ncol(seurat_obj), nrow(seurat_obj)))
    cat(sprintf("Metadata columns: %s\n", 
                paste(names(seurat_obj@meta.data), collapse = ", ")))
    
    # Return success
    list(
        success = TRUE,
        message = sprintf("Successfully converted to Seurat object: %d cells, %d genes", 
                         ncol(seurat_obj), nrow(seurat_obj))
    )
}