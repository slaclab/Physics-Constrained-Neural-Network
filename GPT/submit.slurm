#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=rfar:regular
#SBATCH --job-name=PCNN
#SBATCH --output=/sdf/home/j/jcurcio/GPT/output-%j.txt
#SBATCH --error=/sdf/home/j/jcurcio/GPT/output-%j.txt
#
#SBATCH --nodes=1
#SBATCH --mem=480gb
#
#SBATCH --time=0-08:00:00

module load gpt

mr -v -o PINN_trainingData_08.gdf  UEDbeam3MeV.mr gpt  working.in
