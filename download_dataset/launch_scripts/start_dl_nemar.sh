#!/bin/bash

#SBATCH --job-name=dl_nemar
#SBATCH --output=dl_nemar_output_%j.out
#SBATCH --error=dl_nemar_error_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu_il

python3 dl_nemar.py
