#!/bin/bash

#SBATCH --partition=milano
#
#SBATCH --job-name=PCNN_Parsing
#SBATCH --output=/sdf/home/j/jcurcio/pcnn/Parsing/outputs/output-%j.txt
#SBATCH --error=/sdf/home/j/jcurcio/pcnn/Parsing/outputs/output-%j.txt
#
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12g
#
#SBATCH --time=0-00:20:00

# Run your Python script
python parse.py --parse

