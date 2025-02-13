import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

def postprocess_results():
    """Post-process the deconvolution results"""
    # Load the results from Cell2location
    adata = sc.read_h5ad("0502newsample_c2loutput.h5ad")
    
    # Extract cell abundance results
    cell_abundance = pd.DataFrame(adata.obsm['q05_cell_abundance_w_sf'])
    
    # Filter out low-abundance cells
    cell_abundance = cell_abundance[cell_abundance.max(axis=1) > 0.1]
    
    # Add tissue location information for visualization
    adata.obs['cell_abundance'] = cell_abundance.max(axis=1)

    # Visualize cell abundance across tissue locations
    sc.pl.spatial(adata, color='cell_abundance', cmap='magma', size=1)

    # Save post-processed results
    adata.write("0502postprocessed_results.h5ad")
    print("Post-processed results saved as '0502postprocessed_results.h5ad'.")

if __name__ == "__main__":
    postprocess_results()
