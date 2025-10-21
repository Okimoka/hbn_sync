from pathlib import Path
import os
import fnmatch
import re
import pandas as pd

from .log_utils import log, crash
from .io_utils import read_table_with_sniff, write_table
from .constants import is_subject_dir, ALLOWED_EEG_PATTERNS



RUNLESS_SUFFIXES = ['channels.tsv', 'events.tsv', 'eeg.json', 'eeg.set']

def _subject_id_only(sub_dir_name: str) -> str:
    m = re.match(r"^sub-(.+)$", sub_dir_name)
    return m.group(1) if m else sub_dir_name

def _pattern_matches(filename: str, sub_dir_name: str) -> bool:
    sid = _subject_id_only(sub_dir_name)
    for pat in ALLOWED_EEG_PATTERNS:
        concrete = pat.replace("{SUB}", sub_dir_name).replace("{ID}", sid)
        if fnmatch.fnmatch(filename, concrete):
            return True
    return False

def _insert_run1_filename(name: str) -> str:
    for suf in RUNLESS_SUFFIXES:
        if name.endswith("_" + suf):
            return name[: -len(suf) - 1] + f"_run-1_{suf}"
    crash(f"Cannot insert run-1 for filename without a known suffix: {name}")
    return name

def _ensure_single_eeg_folder(sub_root: Path) -> Path:
    eeg_dirs = [p for p in sub_root.iterdir() if p.is_dir() and p.name == "eeg"]
    if len(eeg_dirs) != 1:
        crash(f"{sub_root} must contain exactly one 'eeg' folder; found {len(eeg_dirs)}")
    return eeg_dirs[0]

def _check_allowlist_and_log(eeg_dir: Path, script_dir: Path):
    if not ALLOWED_EEG_PATTERNS:
        return
    sub_name = eeg_dir.parent.name
    extras = []
    for p in eeg_dir.iterdir():
        if not p.is_file():
            continue
        if not _pattern_matches(p.name, sub_name):
            extras.append(p.name)
    if extras:
        log(script_dir, f"[{sub_name}] unexpected eeg files not allowed by patterns: {sorted(extras)}")

def _fix_channels_file(src_path: Path, dst_path: Path):
    df = read_table_with_sniff(src_path)
    if 'name' not in df.columns:
        crash(f"Missing required 'name' column in channels file: {src_path}")

    if 'type' not in df.columns:
        df.insert(len(df.columns), 'type', 'EEG')
    else:
        df['type'] = 'EEG'
    df.loc[df['name'] == 'Cz', 'type'] = 'MISC'

    if 'units' not in df.columns:
        df.insert(len(df.columns), 'units', 'uV')
    else:
        df['units'] = 'uV'

    write_table(df, dst_path, delimiter='\t')

def _safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.fspath(src), os.fspath(dst))
    except OSError as e:
        crash(f"Failed to create symlink: {dst} -> {src} ({e})")

def process_subjects(release_roots, merged_root: Path, script_dir: Path):
    merged_root = merged_root.resolve()

    for rel_root in release_roots:
        for item in rel_root.iterdir():
            if not item.is_dir() or not is_subject_dir(item.name):
                continue
            sub_id = item.name
            eeg_dir = _ensure_single_eeg_folder(item)

            _check_allowlist_and_log(eeg_dir, script_dir)

            out_sub_eeg = merged_root / sub_id / "eeg"
            out_sub_eeg.mkdir(parents=True, exist_ok=True)

            for f in sorted(p for p in eeg_dir.iterdir() if p.is_file()):
                name = f.name
                has_run = "_run-" in name
                is_channels = name.endswith("_channels.tsv")
                is_runless_suffix = any(name.endswith("_" + s) for s in RUNLESS_SUFFIXES)

                if is_channels:
                    out_name = name if has_run else _insert_run1_filename(name)
                    dst = out_sub_eeg / out_name
                    _fix_channels_file(f, dst)
                else:
                    if not has_run and is_runless_suffix:
                        out_name = _insert_run1_filename(name)
                        dst = out_sub_eeg / out_name
                        _safe_symlink(f, dst)
                    elif has_run:
                        dst = out_sub_eeg / name
                        _safe_symlink(f, dst)
                    else:
                        log(script_dir, f"[{sub_id}] Unhandled file (no _run- and unexpected suffix): {name}")
