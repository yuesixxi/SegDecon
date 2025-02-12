import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import cell2location as c2l
import matplotlib.pyplot as plt
import warnings

# Set parameters
def setup_environment():
    """Setup the environment by installing required packages"""
    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        get_ipython().system('pip install --quiet scvi-colab')
        from scvi_colab import install
        install()
        get_ipython().system('pip install --quiet git+https://github.com/BayraktarLab/cell2location#egg=cell2location[tutorials]')

def load_data():
    """Load the datasets"""
    adata_st = sc.read_h5ad('./0502stardist_nuclei_c2linput.h5ad')
    adata_ref = sc.read('../results/mouse_brain_snrna/regression_model/RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_12819genes/sc.h5ad')
    return adata_st, adata_ref

def preprocess_data(adata_st, adata_ref):
    """Preprocess the data"""
    # Prepare feature names
    adata_st.var["feature_name"] = adata_st.var_names
    adata_st.var.set_index("gene_ids", drop=True, inplace=True)
    
    # Handle mitochondria-encoded genes
    adata_st.var['MT_gene'] = [gene.startswith('mt-') for gene in adata_st.var.index]
    adata_st.obsm['MT'] = adata_st[:, adata_st.var['MT_gene'].values].X.toarray()
    adata_st = adata_st[:, ~adata_st.var['MT_gene'].values]

    return adata_st, adata_ref

def get_shared_features(adata_st, adata_ref):
    """Find and align shared features between spatial and reference data"""
    shared_features = [feature for feature in adata_st.var_names if feature in adata_ref.var_names]
    adata_sc = adata_ref[:, shared_features].copy()
    adata_st = adata_st[:, shared_features].copy()
    return adata_st, adata_sc

def prepare_for_cell2location(adata_st, adata_ref):
    """Prepare data for Cell2location"""
    covariate_col_names = 'annotation_1'
    inf_aver = adata_ref.raw.var.copy()
    inf_aver = inf_aver.loc[:, [f'means_cov_effect_{covariate_col_names}_{i}' for i in adata_ref.obs[covariate_col_names].unique()]]
    inf_aver.columns = [sub(f'means_cov_effect_{covariate_col_names}_{i}', '', i) for i in adata_ref.obs[covariate_col_names].unique()]
    inf_aver = inf_aver.iloc[:, inf_aver.columns.argsort()]
    inf_aver = inf_aver * adata_ref.uns['regression_mod']['post_sample_means']['sample_scaling'].mean()
    return inf_aver

def setup_cell2location_model(adata_st, inf_aver):
    """Setup and initialize the Cell2location model"""
    c2l.models.Cell2location.setup_anndata(adata=adata_st)
    model = c2l.models.Cell2location(
        adata_st,
        cell_state_df=inf_aver,
        N_cells_per_location=float(adata_st.obs['nuclei_count'].mean()),
        N_cells_mean_var_ratio=float(adata_st.obs['nuclei_count'].var())
    )
    return model

def train_and_export(model):
    """Train model and export results"""
    model.train(max_epochs=30000)
    model.plot_history()
    model.export_posterior()
    model.plot_QC()

def save_results(adata_st):
    """Save results"""
    adata_st.write("0502newsample_c2loutput.h5ad")

def main():
    setup_environment()
    adata_st, adata_ref = load_data()
    adata_st, adata_ref = preprocess_data(adata_st, adata_ref)
    shared_features = adata_st, adata_sc = get_shared_features(adata_st, adata_ref)
    inf_aver = prepare_for_cell2location(adata_st, adata_ref)
    model = setup_cell2location_model(adata_st, inf_aver)
    train_and_export(model)
    save_results(adata_st)

if __name__ == "__main__":
    main()

