#!/usr/bin/env python
# coding: utf-8

import os
import tarfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import scanpy as sc
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon
from tifffile import imread
from .kmeans_noise_filter import get_default_noise_thresholds, get_kmeans_noise_thresholds  

def download_and_extract_data():
    """Download and extract all required data into the data/ directory"""
    
    # Define paths
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")  # data/ directory
    extract_path = os.path.join(data_path, "spatial")  # Extracted spatial data path
    tar_path = os.path.join(data_path, "CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_spatial.tar.gz")

    # Create necessary directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    # Download required files directly into the data/ directory
    print("Downloading files into the data/ directory...")
    os.system(f'curl -o {data_path}/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_tissue_image.tif '
              'https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_tissue_image.tif')

    os.system(f'curl -o {tar_path} '
              'https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_spatial.tar.gz')

    os.system(f'curl -o {data_path}/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_filtered_feature_bc_matrix.h5 '
              'https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_filtered_feature_bc_matrix.h5')

    # Extract tar.gz file into data/spatial/
    print(f"Extracting {tar_path} to {extract_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

    print(f"Extraction completed, files extracted to: {extract_path}")

import os
import numpy as np
import cv2
from skimage.io import imread
from .kmeans_noise_filter import get_default_noise_thresholds, get_kmeans_noise_thresholds  # 关键导入

def preprocess_image():
    """Image preprocessing: noise reduction and HSV conversion"""

    # Get the image path
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_tissue_image.tif")

    print(f"Loading image from {img_path}...")
    img = imread(img_path, plugin='tifffile')

    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Use the default noise threshold
    lower_noise, upper_noise = get_default_noise_thresholds()

    # **Optional**: Provide KMeans reference, not used by default
    print("Suggested noise thresholds from KMeans (not used by default):")
    suggested_lower, suggested_upper = get_kmeans_noise_thresholds(img_hsv, crop_region=(17500, 22500, 7000, 12000))
    print(f"Suggested Lower HSV: {suggested_lower}, Suggested Upper HSV: {suggested_upper}")

    # Noise removal
    mask = cv2.inRange(img_hsv, lower_noise, upper_noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) 
    connected_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    replacement_color = np.array([321 / 2, 45 * 2.55, 98 * 2.55], dtype=np.uint8).reshape(1, 1, 3)
    img_hsv[connected_mask > 0] = replacement_color
    modified_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return modified_img

def segment_nuclei(modified_img):
    """Nuclei segmentation using Stardist model"""
    model = StarDist2D.from_pretrained('2D_versatile_he')
    img = normalize(modified_img, 5, 95)

    # Adjust nms_thresh and prob_thresh 
    labels, polys = model.predict_instances_big(
        img, 
        axes='YXC', 
        block_size=4096, 
        prob_thresh=0.2, 
        nms_thresh=0.001,
        min_overlap=128,  
        context=128,       
        normalizer=None, 
        n_tiles=(4,4,1)    
    )

    return polys


def create_geodataframe(polys):
    """Create GeoDataFrame from segmented polygons"""
    geometries = []
    for nuclei in range(len(polys['coord'])):
        coords = [(y, x) for x, y in zip(polys['coord'][nuclei][0], polys['coord'][nuclei][1])]
        geometries.append(Polygon(coords))

    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf['id'] = [f"ID_{i+1}" for i, _ in enumerate(gdf.index)]
    return gdf

def filter_cells(gdf):
    """Filter nuclei with area greater than 35 and less than 1500"""
    # Adjust this range manually according to specific data.
    return gdf[(gdf['area'] > 35) & (gdf['area'] < 1500)]

def process_spatial_data():
    """Read and process spatial transcriptomics data"""
    adata = sc.read_10x_h5('../data/CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_filtered_feature_bc_matrix.h5')

    # QC calculations
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Load spatial coordinates
    df_tissue_positions = pd.read_csv('../data/spatial/spatial/tissue_positions.csv', index_col='barcode')
    adata.obs = adata.obs.merge(df_tissue_positions, left_index=True, right_index=True, how='left')
    assert adata.obs.shape[0] == adata.n_obs, "Number of rows in adata.obs doesn't match expected."

    # Store spatial coordinates in obsm
    spatial_coords = np.array([df_tissue_positions['pxl_col_in_fullres'], df_tissue_positions['pxl_row_in_fullres']]).T
    adata.obsm['spatial'] = spatial_coords
    return adata, df_tissue_positions

def run_pipeline():
    """Run the entire pipeline"""
    download_and_extract_data()
    modified_img = preprocess_image()
    polys = segment_nuclei(modified_img)
    gdf = create_geodataframe(polys)
    gdf_filtered = filter_cells(gdf)

    # Process spatial transcriptomics data
    adata, df_tissue_positions = process_spatial_data()

    # Save results
    adata.write('0502stardist_nuclei_c2linput.h5ad')
    print(f"Processed data saved as '0502stardist_nuclei_c2linput.h5ad'.")

if __name__ == "__main__":
    run_pipeline()
