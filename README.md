## Overview
This repository is an adaptation of Alexander Scheinker's work (cited at the bottom). My goal was to modify his model such that it could take dynamic input resolutions. This is useful because real and simulated data isn't always capable of being discretized to a 128x128x128 grid. This code was run on NERSC's Jupyter hub and SLAC's S3DF, however, I included all of the S3DF slurm scripts for ease of use. For the following instructions I'm assuming access to S3DF.
## Steps to run the code
### 1. Data generation
Note: I included my most recent beam data called **PINN_trainingData_08.gdf** in the Parsing directory

In the GPT directory, type *sbatch submit.slurm* to generate new data. This data should be moved to the Parsing directory before proceeding.
### 2. Parsing


A. Scheinker and R. Pokharel, Physics Constrained 3D Convolutional Neural Networks for Electrodynamics, APL Machine Learning, 2023
