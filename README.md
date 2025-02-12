# Spatial Transcriptomics Pipeline

This repository contains the code for processing, segmenting, and analyzing spatial transcriptomics data using deep learning models like Stardist and deconvolution methods like Cell2location.

## Pipeline Overview

1. **Preprocessing and Segmentation** (`Denoising and segmentation.py`)
   - Download and preprocess tissue images.
   - Perform nuclei segmentation using Stardist.

2. **Deconvolution** (`deconvolution.py`)
   - Perform deconvolution using Cell2location to assign spatial cell abundances.
   - Train a Cell2location model with the spatial transcriptomics data.

3. **Postprocessing** (`postprocess.py`)
   - Post-process the results, visualize cell abundances, and save final data.

## Installation

1. Clone this repository:
git clone https://github.com/yuesixxi/SegDecon.git cd SegDecon

2. Set up the environment using Conda:
conda env create -f environment.yml conda activate SegDecon

## Running the Pipeline

Execute the pipeline by running the following script:
SegDecon/Pipeline/Denoising and segmentation.py 
SegDecon/Pipeline/deconvolution.py 
SefDecon/Pipeline/postprocess.py


## Results

The processed and analyzed data will be saved in the `data/` folder, including:
- `deconvolution_input.h5ad`
- `deconvolution_output.h5ad`
- `segdecon_results.h5ad`


