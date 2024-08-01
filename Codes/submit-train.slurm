#!/bin/bash

#SBATCH --partition=ampere
#
#SBATCH --job-name=PCNN
#SBATCH --output=/sdf/home/j/jcurcio/pcnn/Codes/outputs/output-%j.txt
#SBATCH --error=/sdf/home/j/jcurcio/pcnn/Codes/outputs/output-%j.txt
#
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32g
#
#SBATCH --time=0-02:30:00
#
#SBATCH --gpus 4

# Run your Python script
python Dynamic_PCNN.py
