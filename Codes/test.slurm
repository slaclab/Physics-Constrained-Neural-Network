#!/bin/bash

#SBATCH --partition=ampere
#
#SBATCH --job-name=test
#SBATCH --output=/sdf/home/j/jcurcio/pcnn/Codes/outputs/output-%j.txt
#SBATCH --error=/sdf/home/j/jcurcio/pcnn/Codes/outputs/output-%j.txt
#
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8g
#
#SBATCH --time=0-00:10:00
#
#SBATCH --gpus 4

# Run your Python script
python Test_PCNN.py
