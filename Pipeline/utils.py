import numpy as np
import scanpy as sc

def filter_genes(adata, min_counts=10, min_cells=3):
    """Filter genes based on minimum count and cell expression"""
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    return adata

def normalize_data(adata):
    """Normalize gene expression data"""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata

def preprocess_annotations(adata):
    """Preprocess annotations (e.g., removing unwanted genes or categories)"""
    adata.obs['annotation'] = adata.obs['annotation'].str.replace("unwanted_category", "")
    return adata

def extract_coordinates(adata):
    """Extract spatial coordinates"""
    return adata.obsm['spatial']
