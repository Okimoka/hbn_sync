#!/bin/bash

#SBATCH --job-name=dl_bucket
#SBATCH --output=dl_bucket_output_%j.out
#SBATCH --error=dl_bucket_error_%j.err
#SBATCH --time=2-05:00:00
#SBATCH --partition=cpu_il

python3 dl_bucket.py
