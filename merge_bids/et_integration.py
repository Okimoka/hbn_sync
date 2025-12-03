from pathlib import Path
import os
import re
import fnmatch
from typing import Dict, Optional, Tuple

from .log_utils import log, crash
from .validators import validate_subject_et_structure
from .constants import ET_TO_EEG_TASK

"""
This subject has completely wrong filenames:
NDARRV837BZQ
The filenames are:
5008455_Video4_Events.txt
5008455_Video4_Samples.txt

These subject names are shorter/longer than all others:
NDARJ257ZU2
NDARJH99CWH
NDARXF358XGE1
NDARXL697DA61
NDARAEZ493ZJ6
NDARAM487XU33

This file contains a stray "C" (malformed), just ignore
The correct file is also present without .save suffix
DATASET/ET/NDARJW326UED/txt/NDARJW326UED_resting_Events.txt.save

"""



def _parse_txt_filename(name: str, subject_id: str):
    # All these subjects have an extra number added to their ET filenames
    # e.g. NDARAEZ493ZJ has filenames like "NDARAEZ493ZJ6_WISC_2nd_Events.txt"
    if subject_id in ["NDARAEZ493ZJ", "NDARXF358XGE", "NDARAM487XU3", "NDARXL697DA6", "NDARHC661KGK"]:
        pattern_weird = rf"^{re.escape(subject_id)}.(?P<sep>__|-|_)(?P<task>[A-Za-z0-9_-]+)_(?P<kind>Events|Samples)\.txt$"
        m = re.match(pattern_weird, name)
        if m:
            return m.group("task"), m.group("kind"), m.group("sep")

    # Accept "_", "__", or "-" between subject_id and task
    pattern = rf"^{re.escape(subject_id)}(?P<sep>__|-|_)(?P<task>[A-Za-z0-9_-]+)_(?P<kind>Events|Samples)\.txt$"
    m = re.match(pattern, name)
    if not m:
        crash(f"Unexpected ET TXT filename format for subject {subject_id}: {name}")
    return m.group("task"), m.group("kind"), m.group("sep")  # ettask, kind, separator



#if ET contains "vis_learn", mapping is dependent on what is found in the EEG folder
def _infer_vis_learn_mapping(merged_root: Path, subject_id: str) -> str:
    eeg_dir = merged_root / f"sub-{subject_id}" / "eeg"
    if not eeg_dir.exists() or not eeg_dir.is_dir():
        log(f"Cannot resolve 'vis_learn' mapping for sub-{subject_id}: EEG folder not found at {eeg_dir}")
        return None

    files = [p.name for p in eeg_dir.iterdir() if p.is_file()]
    has6 = any("task-seqLearning6target" in n for n in files)
    has8 = any("task-seqLearning8target" in n for n in files)

    if has6 and has8:
        crash(f"Ambiguous 'vis_learn' mapping for sub-{subject_id}: both seqLearning6target and seqLearning8target present")
    if has6:
        return "seqLearning6target"
    if has8:
        return "seqLearning8target"

    log(f"Cannot map 'vis_learn' for sub-{subject_id}: neither seqLearning6target nor seqLearning8target found in EEG")
    return None



def _symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.fspath(src), os.fspath(dst))
    except OSError as e:
        crash(f"Failed to create symlink: {dst} -> {src} ({e})")

def integrate_et(et_root: Path, merged_root: Path, script_dir: Path):
    et_root = et_root.resolve()
    merged_root = merged_root.resolve()
    if not et_root.exists() or not et_root.is_dir():
        crash(f"ET folder not found at {et_root}")

    #hardcode a subject to start from, comment out check below as well
    #start_subject = "NDARRV688GUX"

    for subj in sorted(p for p in et_root.iterdir() if p.is_dir()):
        subject_id = subj.name  # plain id lke NDARAA075AMK

        # Nothing resembling these subjects exists in the EEG
        if(subject_id in ["NDARJ257ZU2_", "NDARJH99CWH_"]):
            continue

        #TODO These subjects would need specific handling
        if(subject_id in ["NDARRV837BZQ", "NDARJW326UED"]):
            continue

        # Skip until starting subject is reached
        #if subject_id < start_subject:
        #    continue

        # ET data is inside "<id>/Eyetracking"
        # Previous version of this script had them directly inside subject folder
        et_subj_root = subj / "Eyetracking"
        if not et_subj_root.exists() or not et_subj_root.is_dir():
            log(f"Warning: {subject_id} has no Eyetracking folder, skipping")
            continue
            #log(f"Warning: {subject_id} has no Eyetracking folder, using directly contained ET")
            #et_subj_root = subj
    
        misc_subj_root = subj / "Behavioral"
        if misc_subj_root.exists() and misc_subj_root.is_dir():
            misc_dir = merged_root / f"sub-{subject_id}" / "misc"
            misc_dir.mkdir(parents=True, exist_ok=True)
            for item in sorted(misc_subj_root.iterdir()):
                # symlink both files and directories as-is
                dst = misc_dir / item.name
                _symlink(item, dst)
        else:
            log(f"Warning: {subject_id} has no Behavioral folder, skipping misc integration")
    

        validate_subject_et_structure(subj, subject_id)

        # TODO currently only handling txt, tsv likely impossible to analyze
        txt_dir = et_subj_root / "txt"
        if not txt_dir.exists() or not txt_dir.is_dir():
            #log(f"No TXT folder for subject {subject_id} at {txt_dir}, skipping") # too much noise
            continue


        beh_dir = merged_root / f"sub-{subject_id}" / "beh"
        beh_dir.mkdir(parents=True, exist_ok=True)

        for f in sorted(p for p in txt_dir.iterdir() if p.is_file()):
            if f.name.endswith(".save"):
                continue  # ignore stray .save files completely

            ettask, kind, _sep = _parse_txt_filename(f.name, subject_id)

            if ettask == "vis_learn" or ettask == "NODOOR_vis_learn":
                eeg_task = _infer_vis_learn_mapping(merged_root, subject_id)
            else:
                try:
                    eeg_task = ET_TO_EEG_TASK[ettask]
                    if not eeg_task:
                        log(f"No matching EEG task for '{ettask}' (subject {subject_id})")
                        continue
                except KeyError:
                    crash(f"No ET_TO_EEG_TASK mapping provided for ET task '{ettask}' (subject {subject_id})")

            if kind == "Samples":
                out_name = f"sub-{subject_id}_task-{eeg_task}_et.txt"
            else:
                out_name = f"sub-{subject_id}_task-{eeg_task}_et_Events.txt"
            _symlink(f, beh_dir / out_name)
            print("Symlink: " + str(beh_dir / out_name))


    