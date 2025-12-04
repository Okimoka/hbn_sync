This folder contains helper scripts to download and extract the full HBN and HBN-EEG datasets. \
`launch_scripts` contains bash scripts so everything can be ran easily using `sbatch launch_scripts/start_XYZ.sh` on a lustre cluster.

In short: \
`dl_nemar.py` downloads the HBN-EEG zips into nemar_zips \
`dl_bucket.py` downloads the HBN tars into bucket_tars

To verify that everything downloaded properly, there should be
- 11 zips of size 1.7 TiB in nemar_zips
- 4576 .tar.gzs of size 5.6 TiB in bucket_tars

Once this is given, the zips and tars have to be extracted: \
`unpack_nemar.py` unpacks all files from nemar_zips into nemar_zips_unpacked \
`unpack_bucket.py` unpacks all files from bucket_tars into bucket_tars_unpacked \

Once everything is extracted: \
`make_dataset.sh` moves everything into one folder so it's ready to be processed by `create_merged_dataset.py`
