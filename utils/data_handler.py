"""Data handling utilities for AnnData and Seurat objects."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
import anndata as ad
import pandas as pd
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataHandler:
    """Handles loading, saving, and converting single-cell data formats."""
    
    @staticmethod
    def load_data(file_path: Union[str, Path]) -> ad.AnnData:
        """Load data from various formats."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Load based on extension
        if path.suffix in [".h5ad", ".h5"]:
            logger.info(f"Loading AnnData from {path}")
            return ad.read_h5ad(path)
        elif path.suffix == ".csv":
            logger.info(f"Loading CSV data from {path}")
            df = pd.read_csv(path, index_col=0)
            return ad.AnnData(df)
        elif path.suffix == ".txt" or path.suffix == ".tsv":
            logger.info(f"Loading TSV data from {path}")
            df = pd.read_csv(path, sep="\t", index_col=0)
            return ad.AnnData(df)
        elif path.suffix == ".rds":
            raise NotImplementedError("Seurat object loading not yet implemented. Please convert to h5ad format.")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @staticmethod
    def save_data(adata: ad.AnnData, file_path: Union[str, Path]) -> None:
        """Save AnnData object."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving AnnData to {path}")
        adata.write_h5ad(path)
    
    @staticmethod
    def copy_data(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """Copy data file to destination."""
        src_path = Path(src)
        dst_path = Path(dst)
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        logger.info(f"Copied data from {src_path} to {dst_path}")
    
    @staticmethod
    def get_data_summary(adata: ad.AnnData) -> Dict[str, Any]:
        """Get summary statistics of the data."""
        summary = {
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "layers": list(adata.layers.keys()),
            "obsm_keys": list(adata.obsm.keys()),
            "varm_keys": list(adata.varm.keys()),
            "uns_keys": list(adata.uns.keys()),
        }
        
        # Add basic statistics
        if adata.X is not None:
            if hasattr(adata.X, "data"):  # Sparse matrix
                summary["x_mean"] = float(adata.X.data.mean())
                summary["x_std"] = float(adata.X.data.std())
                summary["x_min"] = float(adata.X.data.min())
                summary["x_max"] = float(adata.X.data.max())
                summary["x_sparsity"] = 1.0 - (adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]))
            else:  # Dense matrix
                summary["x_mean"] = float(np.mean(adata.X))
                summary["x_std"] = float(np.std(adata.X))
                summary["x_min"] = float(np.min(adata.X))
                summary["x_max"] = float(np.max(adata.X))
                summary["x_sparsity"] = float(np.sum(adata.X == 0) / adata.X.size)
        
        return summary
    
    @staticmethod
    def decompose_anndata(adata: ad.AnnData, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Decompose AnnData into separate files for container access."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Save main data matrix
        if adata.X is not None:
            x_path = output_path / "X.npz"
            if hasattr(adata.X, "toarray"):  # Sparse matrix
                np.savez_compressed(x_path, data=adata.X.toarray())
            else:
                np.savez_compressed(x_path, data=adata.X)
            files["X"] = str(x_path)
        
        # Save observations
        if not adata.obs.empty:
            obs_path = output_path / "obs.csv"
            adata.obs.to_csv(obs_path)
            files["obs"] = str(obs_path)
        
        # Save variables
        if not adata.var.empty:
            var_path = output_path / "var.csv"
            adata.var.to_csv(var_path)
            files["var"] = str(var_path)
        
        # Save layers
        if adata.layers:
            layers_dir = output_path / "layers"
            layers_dir.mkdir(exist_ok=True)
            for key, layer in adata.layers.items():
                layer_path = layers_dir / f"{key}.npz"
                if hasattr(layer, "toarray"):
                    np.savez_compressed(layer_path, data=layer.toarray())
                else:
                    np.savez_compressed(layer_path, data=layer)
                files[f"layers/{key}"] = str(layer_path)
        
        # Save dimensional reductions
        if adata.obsm:
            obsm_dir = output_path / "obsm"
            obsm_dir.mkdir(exist_ok=True)
            for key, embedding in adata.obsm.items():
                embed_path = obsm_dir / f"{key}.npy"
                np.save(embed_path, embedding)
                files[f"obsm/{key}"] = str(embed_path)
        
        # Save unstructured data as JSON
        if adata.uns:
            uns_path = output_path / "uns.json"
            # Convert numpy types to Python types for JSON serialization
            uns_data = _convert_numpy_types(adata.uns)
            with open(uns_path, "w") as f:
                json.dump(uns_data, f, indent=2)
            files["uns"] = str(uns_path)
        
        # Save metadata
        metadata = {
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "shape": list(adata.shape),
            "files": files
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Decomposed AnnData into {len(files)} files at {output_path}")
        return files


def _convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj