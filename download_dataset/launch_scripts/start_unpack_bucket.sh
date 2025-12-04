#!/bin/bash

#SBATCH --job-name=unpack_bucket
#SBATCH --output=unpack_bucket_output_%j.out
#SBATCH --error=unpack_bucket_error_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu_il

python3 unpack_bucket.py
