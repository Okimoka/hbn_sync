
from argparse import Namespace
from pathlib import Path
from merge_bids.main import run

"""
Expects the following folder structure:

DATASET/
├─ ET/                                # Eye-Tracking
│  ├─ NDARAB977GFB/
│  │  ├─ txt/
│  │  │  ├─ NDARAB977GFB_Video-DM_Events.txt
│  │  │  ├─ NDARAB977GFB_Video-DM_Samples.txt
│  │  │  └─ ...                       # all et files for participant
│  │  └─ ...                          # optionally, tsv or idf folders
│  ├─ <id>/
│  │  └─ txt/
│  │     └─ ...
│  ├─ <id>/
│  │  └─ txt/
│  │     └─ ...
│  └─ ...                             # for all subjects
│
└─ EEG/                               # 11 BIDS releases (randomly named)
   ├─ release1_any_name/
   │  ├─ dataset_description.json
   │  ├─ participants.tsv
   │  ├─ sub-NDARAB977GFB/
   │  │  └─ eeg/
   │  │     ├─ sub-NDARAB977GFB_task-ThePresent_eeg.set
   │  │     ├─ sub-NDARAB977GFB_task-ThePresent_eeg.json
   │  │     ├─ sub-NDARAB977GFB_task-ThePresent_channels.tsv
   │  │     └─ sub-NDARAB977GFB_task-ThePresent_events.tsv
   │  │     └─ ...
   │  └─ ...                          # for all subjects in this release
   └─ release2/
      └─ ...

"""

args = Namespace(
    dataset=str(Path("./DATASET")),
    output=str(Path("./mergedDataset"))
)
run(args)