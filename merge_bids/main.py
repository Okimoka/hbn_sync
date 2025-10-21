#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

from .constants import PARTICIPANTS_COLUMNS, REQUIRED_ROOT_JSON, TEMPLATE_README
from .io_utils import read_table_with_sniff, write_table, read_json, write_json, read_text, write_text
from .validators import validate_participants_table, find_unexpected_root_entries, ensure_required_root_json
from .mergers import pick_first_and_log_differences, concatenate_changes, merge_quality_tables
from .log_utils import log, crash
from .subjects import process_subjects
from .et_integration import integrate_et

def discover_releases(eeg_root: Path) -> List[Path]:
    releases = [p for p in eeg_root.iterdir() if p.is_dir()]
    if len(releases) != 11:
        log(f"Expected 11 release folders in EEG, found {len(releases)} at {eeg_root}")
    return sorted(releases)

# Merge participants.tsv files
def collect_participants_and_release_numbers(release_roots: List[Path]) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    merged_rows = []
    release_by_label = {}
    for root in release_roots:
        pfile = root / "participants.tsv"
        df = validate_participants_table(pfile)
        release_number = df["release_number"].iloc[0]
        release_by_label[release_number] = root
        # For traceability, keep the source release folder name too
        df = df.copy()
        df.insert(0, "source_release", root.name)
        merged_rows.append(df)
    merged = pd.concat(merged_rows, axis=0, ignore_index=True)
    return merged, release_by_label


def find_release_number_map(release_roots: List[Path]) -> Dict[str, Path]:
    mapping = {}
    for root in release_roots:
        df = read_table_with_sniff(root / "participants.tsv")
        rel = df["release_number"].iloc[0]
        mapping[rel] = root
    return mapping

def ensure_all_required_json_and_choose(release_roots: List[Path], script_dir: Path) -> Dict[str, dict]:
    rel_map = find_release_number_map(release_roots)
    # R1 is used as default fallback
    if "R1" not in rel_map:
        crash("Could not locate Release 1 (release_number == 'R1') from participants.tsv files.")
    chosen_root = rel_map["R1"]

    # dataset_description.json
    # just pick the first one (they are all the same)
    obj = read_json(release_roots[0] / "dataset_description.json")
    obj["Name"] = "Healthy Brain Network (HBN) EEG - Merged Release"

    chosen_dd = obj


    # Other required root-level JSONs
    chosen_others = {}
    for name in REQUIRED_ROOT_JSON:
        objs = []
        for root in release_roots:
            p = root / name
            obj = read_json(p)  # will crash if missing
            objs.append((root.name, obj))
        chosen_obj = pick_first_and_log_differences(objs, lambda m: log(f"{name}: {m}"))
        chosen_others[name] = chosen_obj

    return {"dataset_description.json": chosen_dd, **chosen_others, "chosen_root_for_jsons": chosen_root}


def build_changes(release_roots: List[Path], script_dir: Path) -> str:
    items = []
    for root in release_roots:
        # release number from participants.tsv
        df = read_table_with_sniff(root / "participants.tsv")
        rel = df["release_number"].iloc[0]

        changes_path = root / "CHANGES"
        if changes_path.exists():
            txt = read_text(changes_path)
            items.append((rel, txt))
        else:
            # Log and continue if a release has no CHANGES file
            log(f"[{root.name}] No CHANGES file found; skipping.")

    return concatenate_changes(items)


def log_unexpected_entries(release_roots: List[Path], script_dir: Path):
    for root in release_roots:
        unexpected = find_unexpected_root_entries(root)
        if unexpected:
            log(f"[{root.name}] Unexpected root entries: {unexpected}")

def write_merged_root(merged_root: Path, participants_df: pd.DataFrame, chosen_jsons: Dict[str, dict], changes_text: str):
    merged_root.mkdir(parents=True, exist_ok=True)
    # participants.tsv
    write_table(participants_df, merged_root / "participants.tsv")

    # dataset_description.json
    write_json(chosen_jsons["dataset_description.json"], merged_root / "dataset_description.json")

    # CHANGES & README
    write_text(changes_text, merged_root / "CHANGES")
    write_text(TEMPLATE_README, merged_root / "README")


    # Other required JSONs (take from R1/first)
    for name, obj in chosen_jsons.items():
        if name in ("dataset_description.json", "chosen_root_for_jsons"):
            continue
        write_json(obj, merged_root / name)

def run(args):

    log(f"Started merging process")

    script_dir = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset).resolve()
    eeg_root = dataset_root / "EEG"
    if not eeg_root.exists():
        crash(f"EEG folder not found at {eeg_root}")
    releases = discover_releases(eeg_root)

    # Validate & merge participants, fetch release numbers
    participants_df, rel_map = collect_participants_and_release_numbers(releases)

    # Required root JSONs present? (will crash if missing in any release)
    for root in releases:
        _ = ensure_required_root_json(root)

    # Build merged root-level artifacts
    chosen_jsons = ensure_all_required_json_and_choose(releases, script_dir)
    changes_text = build_changes(releases, script_dir)

    # Log any unexpected files/folders in each release root
    log_unexpected_entries(releases, script_dir)

    # Write merged root
    out_root = Path(args.output).resolve()
    
    # to skip some steps, comment them out here
    write_merged_root(out_root, participants_df, chosen_jsons, changes_text)

    # Merge "code" directory quality tables
    merge_quality_tables(releases, out_root / "code", lambda m: log(m))

    print(f"Merged root-level files written to: {out_root}")
    log(f"Merged root-level files written to: {out_root}")

    process_subjects(releases, out_root, script_dir)
    print(f"Subjects written to: {out_root}")
    log(f"Subjects written to: {out_root}")

    
    integrate_et(dataset_root / "ET", out_root, script_dir)
    print(f"ET Merged into: {out_root}")
    log(f"ET Merged into: {out_root}")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple EEG BIDS releases into a single dataset.")
    parser.add_argument("--dataset", required=True, help="Path to the 'dataset' folder containing 'EEG' and 'ET' directories.")
    parser.add_argument("--output", required=True, help="Path to output merged dataset root (e.g., 'merged_dataset').")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
