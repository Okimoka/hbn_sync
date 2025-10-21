from pathlib import Path
import re
from typing import Dict, List, Tuple
import pandas as pd

from .constants import PARTICIPANTS_COLUMNS, REQUIRED_ROOT_JSON, REQUIRED_CODE_TSV, QUALITY_TABLE_COLUMNS, ALLOWED_ROOT_BASENAMES, is_subject_dir, ET_SUBDIRS_ALLOWED
from .io_utils import read_table_with_sniff
from .log_utils import crash

def validate_participants_table(path: Path) -> pd.DataFrame:
    df = read_table_with_sniff(path)
    if list(df.columns) != PARTICIPANTS_COLUMNS:
        crash(f"participants.tsv has incorrect columns at {path}.\nExpected: {PARTICIPANTS_COLUMNS}\nFound: {list(df.columns)}")
    # release_number uniformity check
    uniques = set(df["release_number"].unique())
    if len(uniques) != 1:
        crash(f"Column 'release_number' must have a single unique value in {path}, found {sorted(uniques)}")
    return df

#to enforce strict matching - doesnt work
"""
def validate_code_directory(code_dir: Path) -> Dict[str, Path]:
    if not code_dir.exists() or not code_dir.is_dir():
        crash(f"Missing 'code' directory at {code_dir}")
    files = {p.name: p for p in code_dir.iterdir() if p.is_file()}
    expected = set(REQUIRED_CODE_TSV)
    if set(files.keys()) != expected:
        missing = sorted(expected - set(files.keys()))
        extra = sorted(set(files.keys()) - expected)
        crash(f"'code' directory contents mismatch at {code_dir}.\nMissing: {missing}\nExtra: {extra}")
    return files
"""

def validate_code_directory(code_dir: Path) -> Dict[str, Path]:
    if not code_dir.exists() or not code_dir.is_dir():
        crash(f"Missing 'code' directory at {code_dir}")
    return {p.name: p for p in code_dir.iterdir() if p.is_file()}


def validate_quality_table_columns(path: Path) -> pd.DataFrame:
    df = read_table_with_sniff(path)
    if list(df.columns) != QUALITY_TABLE_COLUMNS:
        crash(f"Quality table has incorrect columns at {path}.\nExpected: {QUALITY_TABLE_COLUMNS}\nFound: {list(df.columns)}")
    return df

def find_unexpected_root_entries(release_root: Path) -> List[str]:
    unexpected = []
    for item in release_root.iterdir():
        name = item.name
        if name in ALLOWED_ROOT_BASENAMES or is_subject_dir(name):
            continue
        unexpected.append(name)
    return sorted(unexpected)

def ensure_required_root_json(release_root: Path) -> List[Path]:
    missing = []
    paths = []
    for name in REQUIRED_ROOT_JSON:
        p = release_root / name
        if not p.exists():
            missing.append(name)
        paths.append(p)
    if missing:
        crash(f"Missing required root JSON files in {release_root}: {missing}")
    return paths


def validate_subject_et_structure(subj_root: Path, subject_id: str):
    # No files at root
    for p in subj_root.iterdir():
        if p.is_file():
            crash(f"Found unexpected file in ET subject root {subj_root}: {p.name}")
    # Only allowed subdirs
    subdirs = [p for p in subj_root.iterdir() if p.is_dir()]
    names = {p.name for p in subdirs}
    if not names.issubset(ET_SUBDIRS_ALLOWED):
        extras = sorted(names - ET_SUBDIRS_ALLOWED)
        crash(f"Unexpected subdirectories in {subj_root}: {extras} (allowed: {ET_SUBDIRS_ALLOWED})")

    # idf must be empty if present
    idf = subj_root / "idf"
    if idf.exists():
        files = [p for p in idf.iterdir() if p.is_file()]
        if files:
            crash(f"'idf' folder must be empty in {subj_root}; found files: {[f.name for f in files]}")

    #TODO currently doesnt recognize blocks
    #just use regex from et_integration directly
    #verification is not really necessary anyway
    #instead of checking for .save, it could just skip the subject
    """
    # tsv names: <id>_<ettaskname>[_(Block|Session)<n>].tsv
    tsv = subj_root / "tsv"
    if tsv.exists():
        for f in tsv.iterdir():
            if f.name.endswith(".save"):
                continue  # ignore singular artifact file
            if not f.is_file():
                continue
            if not re.match(rf"^{re.escape(subject_id)}_[A-Za-z0-9_-]+(?:_(?:Block|Session)\d+)?\.tsv$", f.name):
                crash(
                    f"Invalid ET TSV filename in {tsv}: {f.name} "
                    f"(expected: {subject_id}_<ettaskname>[_(Block|Session)<n>].tsv)"
                )

    # txt names: <id>_<ettaskname>[_(Block|Session)<n>]_(Events|Samples).txt
    txt = subj_root / "txt"
    if txt.exists():
        for f in txt.iterdir():
            if not f.is_file():
                continue
            if not re.match(rf"^{re.escape(subject_id)}_[A-Za-z0-9_-]+(?:_(?:Block|Session)\d+)?_(Events|Samples)\.txt$", f.name):
                crash(
                    f"Invalid ET TXT filename in {txt}: {f.name} "
                    f"(expected: {subject_id}_<ettaskname>[_(Block|Session)<n>]_(Events|Samples).txt)"
                )
    """