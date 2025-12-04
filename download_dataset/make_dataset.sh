#!/usr/bin/env bash

# Create DATASET folder in the current directory and add two symlinks:
#   DATASET/ET  -> ../bucket_tars_unpacked
#   DATASET/EEG -> ../nemar_zips_unpacked

set -euo pipefail

mkdir -p DATASET

ln -sfn ../bucket_tars_unpacked DATASET/ET
ln -sfn ../nemar_zips_unpacked DATASET/EEG
