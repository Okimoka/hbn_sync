#!/bin/bash

#SBATCH --job-name=unpack_nemar
#SBATCH --output=unpack_nemar_output_%j.out
#SBATCH --error=unpack_nemar_error_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu_il

python3 unpack_nemar.py
