# Additional R libraries installation
# Based on ragene-R-job install scripts

# Try to install additional packages that might fail
tryCatch({
  # Additional visualization
  install.packages(c("ggridges", "ggforce", "ggdendro", "ggalluvial"))
  
  # Additional analysis packages
  install.packages(c("RANN", "fitdistrplus", "lmtest", "MASS", "mgcv"))
  
  # Network analysis
  install.packages(c("igraph", "network", "qgraph"))
  
  # Machine learning
  install.packages(c("ranger", "xgboost", "glmnet"))
  
  # Additional Bioconductor packages
  BiocManager::install(c("muscat", "MAST", "zinbwave", "splatter"))
  
  # Cell-cell communication
  devtools::install_github("sqjin/CellChat")
  devtools::install_github("Teichlab/cellphonedb")
  
  # RNA velocity
  devtools::install_github("velocyto-team/velocyto.R")
  
  print("Additional packages installed successfully")
}, error = function(e) {
  print(paste("Some packages failed to install:", e$message))
})