from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import itertools
from .io_utils import read_json, write_json, read_text, write_text, read_table_with_sniff, write_table
from .validators import validate_code_directory, validate_quality_table_columns
from .constants import REQUIRED_CODE_TSV

# LLM Code

def pick_first_and_log_differences(objs: List[Tuple[str, dict]], logger) -> dict:
    """
    objs: list of (label, dict) pairs. Compares all dicts; logs differences if any pair differs.
    Returns the first dict.
    """
    if not objs:
        return {}
    first_label, first_obj = objs[0]
    for (la, oa), (lb, ob) in itertools.combinations(objs, 2):
        if oa != ob:
            logger(f"Found differing JSON between {la} and {lb}. Using {first_label} for merged.")
            break
    return first_obj

def concatenate_changes(changes_list: List[Tuple[str, str]]) -> str:
    """
    changes_list: list of (release_number, changes_text)
    """
    parts = []
    for rel, txt in changes_list:
        header = f"Changes for Release {rel}"
        parts.append(f"{header}\n{'=' * len(header)}\n")
        parts.append(txt.rstrip() + "\n\n")
    return "".join(parts)


def merge_quality_tables(release_roots: List[Path], out_code_dir: Path, logger) -> None:

    out_code_dir.mkdir(parents=True, exist_ok=True)
    # Validate and then merge each TSV across releases
    for fname in REQUIRED_CODE_TSV:
        dfs = []
        for root in release_roots:
            code_dir = root / "code"
            _ = validate_code_directory(code_dir)  # only ensures dir exists
            path = code_dir / fname
            if not path.exists():
                logger(f"[{root.name}] missing code table '{fname}'; skipping this release.")
                continue
            df = validate_quality_table_columns(path)
            # add source release info for traceability
            df = df.copy()
            df.insert(0, "source_release", root.name)
            dfs.append(df)

        if not dfs:
            logger(f"No sources found for code table '{fname}' across all releases; not writing this file.")
            continue

        merged = pd.concat(dfs, axis=0, ignore_index=True)
        write_table(merged, out_code_dir / fname)
        #logger(f"Merged code table: {fname} -> {out_code_dir / fname}")