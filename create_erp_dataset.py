import os
import shutil
import pandas as pd
import re

"""
This script serves as a helper to bring the dataset into a format suitable for mne-bids-pipeline branch for ET syncing.
The original BIDS dataset is expected to be in the directory ./ds005516 (relative to the script) (this corresponds to HBN-EEG release 11 on NeMO/NeuroBridge).
The ET is expected in ./ds005516_ET with a folder for each subject
The new BIDS dataset is saved in ./extractedDataset

Current behavior:
- Extract/copy only the following single-run tasks (no _run-<n> in filenames):
    DespicableMe, DiaryOfAWimpyKid, FunwithFractals, ThePresent, symbolSearch
- Only copy a task for a subject if participants.tsv marks that task as "available".
- Copy root-level task jsons, participants files, dataset description, README, CHANGES, etc.
- Fix *_channels.tsv files to tag channels as EEG (and Cz as MISC), set units to uV.

- EEG files are copied with '_run-1' inserted before their suffix.
- Eye tracking data is copied into a sibling 'beh' folder and renamed 

Expected inputs:
- Original BIDS dataset: ./ds005516   (HBN-EEG release 11)
- Eye tracking dataset:  ./ds005516_ET/<subjectid>/Eyetracking/(txt|tsv|<whatever>)/<files>
- Output dataset:        ./extractedDataset

Code written mostly by LLM
"""

TASKS = [
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
    "ThePresent",
    "symbolSearch",
]

TASKS_ET = [
    "Video-DM",
    "Video-WK",
    "Video-FF",
    "Video-TP",
    "WISC_ProcSpeed",
]

# Map BIDS task -> ET code used in filenames
TASK_TO_ET = dict(zip(TASKS, TASKS_ET))



def copy_eye_tracking_to_beh(subject_id,
                             available_tasks,
                             et_root_dir='ds005516_ET',
                             dest_root_dir='extractedDataset'):
    """
    Copy ET files for a subject into 'beh' next to 'eeg', then rename Samples/Events.

    - Searches ET at:
        ./ds005516_ET/<subjectid>/Eyetracking/*
    - Copies contents of any immediate subfolders (txt/, tsv/, or others).
    - Renames the recognized Samples/Events files using TASK_TO_ET.

    Final names:
        sub-<ID>_task-<task>_et.txt
        sub-<ID>_task-<task>_et_Events.txt
    """
    print(f"  - ET: Preparing to copy eye tracking for {subject_id}...")
    subject_plain = subject_id.replace('sub-', '')
    dest_beh_dir = os.path.join(dest_root_dir, subject_id, 'beh')
    os.makedirs(dest_beh_dir, exist_ok=True)

    # Find the best-matching ET Eyetracking base #TODO simplify this
    p = os.path.join(et_root_dir, subject_plain, 'Eyetracking')
    print(p)
    et_base = p if os.path.isdir(p) else None

    if et_base is None:
        print(f"\033[93m  - Warning: No ET Eyetracking folder found for {subject_id}. Skipping ET.\033[0m")
        return

    # Identify immediate subfolders (txt/, tsv/, etc.). If none, copy files from et_base.
    subdirs = [d for d in os.listdir(et_base) if os.path.isdir(os.path.join(et_base, d))]
    source_dirs = []
    if subdirs:
        # Prefer typical folders first, but include all.
        prioritized = sorted(subdirs, key=lambda x: {'txt': 0, 'tsv': 1}.get(x.lower(), 2))
        source_dirs = [os.path.join(et_base, d) for d in prioritized]
    else:
        source_dirs = [et_base]

    # Copy all files from chosen source dirs into beh (flat).
    copied_files = []
    for src_dir in source_dirs:
        for name in os.listdir(src_dir):
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                dst = os.path.join(dest_beh_dir, name)
                try:
                    shutil.copy2(src, dst)
                    copied_files.append(name)
                except Exception as e:
                    print(f"\033[93m    - Warning: Failed to copy {src}: {e}\033[0m")

    if not copied_files:
        print(f"\033[93m  - Warning: No ET files found to copy for {subject_id}.\033[0m")
        return

    print(f"    - Copied {len(copied_files)} ET file(s) into {dest_beh_dir}")

    # Rename Samples/Events files for tasks actually available for this subject
    files_in_beh = os.listdir(dest_beh_dir)

    for task in available_tasks:
        et_code = TASK_TO_ET.get(task)
        if not et_code:
            continue

        # Build regex (case-insensitive) for Samples and Events
        # Examples to match:
        #   NDARAB678VYW_WISC_ProcSpeed_Samples.txt
        #   NDARAB678VYW_WISC_ProcSpeed_Samples.tsv
        #   NDARAB678VYW_WISC_ProcSpeed_Events.txt/tsv
        base_prefix = re.escape(subject_plain) + "_" + re.escape(et_code)
        samples_re = re.compile(rf"^{base_prefix}_Samples\.[A-Za-z0-9]+$", re.IGNORECASE)
        events_re  = re.compile(rf"^{base_prefix}_Events\.[A-Za-z0-9]+$", re.IGNORECASE)

        # Find first match for each
        samples_src = next((f for f in files_in_beh if samples_re.match(f)), None)
        events_src  = next((f for f in files_in_beh if events_re.match(f)), None)

        # to retain file endings
        # just force .txt for now, and detect file ending in sync step
        #samples_ext = Path(samples_src).suffix if samples_src else ".txt"
        #events_ext  = Path(events_src).suffix if events_src else ".txt"

        # Prepare target names with .txt extension
        samples_dst = f"{subject_id}_task-{task}_et.txt"
        events_dst  = f"{subject_id}_task-{task}_et_Events.txt"

        # Rename if present
        if samples_src:
            try:
                os.replace(
                    os.path.join(dest_beh_dir, samples_src),
                    os.path.join(dest_beh_dir, samples_dst)
                )
                print(f"    - Renamed ET Samples → {samples_dst}")
            except Exception as e:
                print(f"\033[93m    - Warning: Could not rename {samples_src} → {samples_dst}: {e}\033[0m")
        else:
            print(f"\033[93m    - Warning: Samples file not found for task {task} (pattern {subject_plain}_{et_code}_Samples.*)\033[0m")

        if events_src:
            try:
                os.replace(
                    os.path.join(dest_beh_dir, events_src),
                    os.path.join(dest_beh_dir, events_dst)
                )
                print(f"    - Renamed ET Events  → {events_dst}")
            except Exception as e:
                print(f"\033[93m    - Warning: Could not rename {events_src} → {events_dst}: {e}\033[0m")
        else:
            print(f"\033[93m    - Warning: Events file not found for task {task} (pattern {subject_plain}_{et_code}_Events.*)\033[0m")









def create_bids_subset(source_dir='ds005516', dest_dir='extractedDataset', et_root_dir='ds005516_ET'):

    # --- 1. Define Source and Destination Paths ---
    print(f"Source BIDS dataset: '{source_dir}'")
    print(f"Destination BIDS dataset: '{dest_dir}'")

    if not os.path.isdir(source_dir):
        print(f"\nError: Source directory '{source_dir}' not found.")
        return False

    # --- 2. Create Destination Directory ---
    print(f"\n--- Part 1: Creating BIDS Subset ---")
    print(f"Creating destination directory: '{dest_dir}'...")
    os.makedirs(dest_dir, exist_ok=True)
    # --- 3. Copy Root-Level Files and Directories ---
    print("Copying specified root-level files and directories...")

    # Root-level files common to the dataset
    files_to_copy = [
        'participants.json',
        'participants.tsv',
        'dataset_description.json',
        'README',
        'CHANGES',
        '.gitattributes'
    ]

    # Add task-level json files for each requested task (if they exist)
    # Notably this only works because the specified tasks are single-run tasks
    for task in TASKS:
        files_to_copy.append(f'task-{task}_eeg.json')
        files_to_copy.append(f'task-{task}_events.json')

    dirs_to_copy = [
        '.git',
        '.datalad',
        'code'
    ]

    # Copy files
    for file_name in files_to_copy:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        if os.path.exists(source_path):
            # Make sure destination subdirs exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(source_path, dest_path)
            print(f"  - Copied file: {file_name}")
        else:
            print(f"{'\033[93m'} - Warning: Source file not found, skipping: {source_path} {'\033[0m'}")

    # Copy directories
    for dir_name in dirs_to_copy:
        source_path = os.path.join(source_dir, dir_name)
        dest_path = os.path.join(dest_dir, dir_name)
        if os.path.isdir(source_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path, symlinks=True)
            print(f"  - Copied directory: {dir_name}")
        else:
            print(f"{'\033[93m'} - Warning: Source directory not found, skipping: {source_path} {'\033[0m'}")

    # --- 4. Identify and Process Eligible Subjects ---
    print("\nReading 'participants.tsv' to select subjects and tasks...")
    try:
        participants_df = pd.read_csv(os.path.join(source_dir, 'participants.tsv'), sep='\t')
    except FileNotFoundError:
        print(f"  - Error: 'participants.tsv' not found. Cannot select subject data.")
        return False

    total_subjects = 0
    subjects_with_any_task = 0

    for _, row in participants_df.iterrows():
        subject_id = row.get('participant_id', None)
        if not isinstance(subject_id, str) or not subject_id:
            print(f"{'\033[93m'} - Warning: Skipping a row without a valid 'participant_id'. {'\033[0m'}")

            continue

        available_tasks = []
        for task in TASKS:
            if task in participants_df.columns:
                try:
                    if str(row[task]).strip().lower() == 'available':
                        available_tasks.append(task)
                except KeyError:
                    pass  # column missing for this row; continue
            else:
                # Column missing entirely; warn once per subject
                print(f"{'\033[93m'}  - Warning: Column '{task}' not found in participants.tsv (subject {subject_id}). {'\033[0m'}")

        if not available_tasks:
            continue


        if available_tasks:
            subjects_with_any_task += 1
            print(f"\nProcessing {subject_id}: Found available tasks {available_tasks}")

            source_subject_eeg_dir = os.path.join(source_dir, subject_id, 'eeg')
            dest_subject_eeg_dir = os.path.join(dest_dir, subject_id, 'eeg')

            # Only create the destination subject/eeg folder if we will actually copy something
            os.makedirs(dest_subject_eeg_dir, exist_ok=True)
            
            for task in available_tasks:
                file_base = f"{subject_id}_task-{task}"
                file_suffixes = ['_channels.tsv', '_events.tsv', '_eeg.json', '_eeg.set']

                print(f"  - Copying files for task '{task}' with '_run-1'...")
                for suffix in file_suffixes:
                    src_name = file_base + suffix
                    src_file = os.path.join(source_subject_eeg_dir, src_name)

                    # Insert _run-1 before suffix for destination name
                    dst_name = f"{file_base}_run-1{suffix}"
                    dst_file = os.path.join(dest_subject_eeg_dir, dst_name)

                    if os.path.exists(src_file):
                        try:
                            shutil.copy2(src_file, dst_file)
                            total_subjects += 1  # counting copied file occurrences; optional
                        except Exception as e:
                            print(f"\033[93m    - Warning: Could not copy {src_file} to {dst_file}: {e}\033[0m")
                    else:
                        print(f"\033[93m    - Warning: Source EEG file not found, skipping: {src_file} \033[0m")

            copy_eye_tracking_to_beh(
                subject_id=subject_id,
                available_tasks=available_tasks,
                et_root_dir=et_root_dir,
                dest_root_dir=dest_dir
            )


    print(f"\nSubset creation complete. Subjects with ≥1 available task: {subjects_with_any_task}.")
    return True


def fix_channels_files(dataset_dir='extractedDataset'):
    """
    Fixes *_channels.tsv files in the BIDS dataset:
    - Ensures 'type' column exists; sets default to 'EEG', but 'Cz' (exact match) to 'MISC'.
    - Ensures 'units' column exists and sets to 'uV'.
    NOTE: Does NOT touch *_events.tsv files.
    """
    print(f"\n--- Part 2: Fixing channel metadata in '{dataset_dir}' ---")

    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found. Cannot fix channel files.")
        return

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("_channels.tsv"):
                file_path = os.path.join(root, file)
                print(f"Processing channel file: {file_path}")
                try:
                    channels_df = pd.read_csv(file_path, sep='\t')

                    # Ensure required columns exist
                    if 'name' not in channels_df.columns:
                        print(f"{'\033[93m'} - Warning: 'name' column not found; cannot tag Cz as MISC. {'\033[0m'}")
                    if 'type' not in channels_df.columns:
                        channels_df['type'] = 'EEG'
                    else:
                        channels_df['type'] = 'EEG'  # default

                    if 'name' in channels_df.columns:
                        channels_df.loc[channels_df['name'] == 'Cz', 'type'] = 'MISC'
                        print(f"  - Set 'type' to 'EEG' (or 'MISC' for Cz).")

                    if 'units' not in channels_df.columns:
                        channels_df['units'] = 'uV'
                    else:
                        channels_df['units'] = 'uV'
                    print(f"  - Set 'units' column to 'uV'.")

                    channels_df.to_csv(file_path, sep='\t', index=False)

                except Exception as e:
                    print(f"  - Error processing file {file_path}: {e}")

    print("\nChannel metadata fix complete.")


if __name__ == '__main__':
    source_dataset = 'ds005516'
    new_dataset = 'extractedDataset'
    et_dataset = 'ds005516_ET'

    # Part 1: Create the subset of the BIDS dataset
    success = create_bids_subset(source_dir=source_dataset,
                                 dest_dir=new_dataset,
                                 et_root_dir=et_dataset)
    # Part 2: If the subset was created successfully, fix channels.tsv files
    if success:
        fix_channels_files(dataset_dir=new_dataset)
