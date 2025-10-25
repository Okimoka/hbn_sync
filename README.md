
# hbn_sync
This project aims to curate and synchronize EEG and Eye-Tracking data from the [HBN-EEG dataset](https://neuromechanist.github.io/data/hbn/). To do this, it uses [an mne-bids-pipeline fork](https://github.com/s-ccs/mne-bids-pipeline/tree/temp_dev) and captures metrics about its synchronization quality.


# Instructions

- Download all 11 releases of the HBN-EEG dataset ([nemar](https://nemar.org/dataexplorer/local?search=HBN-EEG))
- Download the accompanying Eye-Tracking data ([all download pages](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/downloads/)). This requires downloading the full dataset, and optionally stripping out all "EEG" and "Behavioral" data

The exact expected folder structure for this data is described in `create_merged_dataset.py`, which is a script that transforms the downloaded datasets into one combined BIDS dataset (`mergedDataset`) which is mne-bids-pipeline compatible.

In order to parse the SMI Eye-Tracker format used in the HBN dataset, a modification to the mne-bids-pipeline has to be made. This repository features two variants of this modification:

- `_05b_sync_eyelink.py` which adds support for the SMI format, plus captures some important metrics about the synchronization quality and stores them in `extractedDataset/derivs/<subject>/eeg/<subject>_task-<task>-eyelink_metrics.json`. This includes some HBN-specific code, like reading the `participants.tsv` from the dataset
- `_05b_sync_eyelink_nometrics.py` which only adds support for the SMI format and does not include any code that would be specific to only this project

After installing the mne-bids-pipeline fork (`pip3 install git+https://github.com/s-ccs/mne-bids-pipeline.git@temp_dev`), the `_05b_sync_eyelink.py` file has to be updated with the first of these variants (i.e. overwrite the file in `~/.local/lib/python3.12/site-packages/mne_bids_pipeline/steps/preprocessing`).

To run all processing steps up to the synchronization step, execute:

```
mne_bids_pipeline --config=config.py --steps init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter.py,preprocessing/_05b_sync_eyelink.py
```

After this step, all subjects with valid EEG and Eye-Tracking data should be populated with their respective `*-eyelink_metrics.json` files. To summarize all these files, run `python3 create_overview_from_derivs.py`, which outputs an overview file inside `extractedDataset/derivs`. A sample output is provided in `sync_metrics_overview.xlsx`.
