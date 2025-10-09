#!/usr/bin/env python3
"""

Scans a derivatives tree for files like:
./extractedDataset/derivs/sub-*/eeg/*_proc-eyelink_metrics.json
and writes an Excel summary with one row per JSON.

Usage:
  python create_overview_from_derivs.py \
    --derivs-root ./extractedDataset/derivs \
    --out sync_metrics_overview.xlsx

Code written mostly by LLM
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Dict, Any, List, Tuple, Iterable


BIDS_ENTITIES = ("sub", "ses", "task", "acq", "rec", "run", "proc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create an Excel overview from *_proc-eyelink_metrics.json derivatives."
    )
    p.add_argument(
        "--derivs-root",
        type=Path,
        default=Path("./extractedDataset/derivs"),
        help="Path to derivatives root (default: ./extractedDataset/derivs)",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*_proc-eyelink_metrics.json",
        help="Filename pattern to match (default: *_proc-eyelink_metrics.json)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("sync_metrics_overview.xlsx"),
        help="Output Excel filename (default: sync_metrics_overview.xlsx). "
             "If relative, it will be saved under --derivs-root.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print each file as it is processed.",
    )
    return p.parse_args()


def rglob_metrics(derivs_root: Path, pattern: str) -> List[Path]:
    # Search recursively; common layout is sub-*/eeg/*.json but EEG level is not hardcoded
    return sorted(derivs_root.rglob(pattern))


def parse_bids_from_path(path: Path) -> Dict[str, Any]:
    """
    Extract BIDS-like entities from the path or filename, if they exist
    """
    text = str(path)
    out = {k: "" for k in BIDS_ENTITIES}
    for ent in BIDS_ENTITIES:
        m = re.search(rf"{ent}-([a-zA-Z0-9]+)", text)
        if m:
            out[ent] = m.group(1)
    return out


def ensure_json_scalars(obj: Any) -> Any:
    """
    Convert numpy-like or non-serializable objects to plain Python scalars/lists.
    """
    try:
        import numpy as np  # optional
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # convert NaN/inf to None for DataFrame consistency
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def load_metrics_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize in case anything odd snuck in
    return {k: ensure_json_scalars(v) for k, v in data.items()}


def union_keys(dicts: Iterable[Dict[str, Any]]) -> List[str]:
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    return sorted(keys)


def build_rows(files: List[Path], verbose: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    for p in files:
        if verbose:
            print(f"Reading {p}")
        try:
            metric = load_metrics_json(p)
        except Exception as e:
            print(f"[WARN] Skipping {p} (failed to read/parse JSON): {e}")
            continue

        meta = parse_bids_from_path(p)
        row = {
            "file_path": str(p),
            **meta,
            **metric,  # JSON fields override meta if names collide (they shouldn't)
        }
        rows.append(row)

    # Decide on a reasonable column order:
    # 1) path + BIDS columns
    # 2) common metric columns if present
    # 3) any extras (sorted)
    # TODO Adjust order to work with new metrics (these are old from eye-eeg)
    preferred_metrics = [
        "eeg_samples", "et_samples",
        "eeg_sampling_rate_hz", "et_sampling_rate_hz",
        "eeg_samples_trimmed", "et_samples_trimmed",
        "et_nan_values",
        "shared_events_pre", "shared_events_post",
        "mean_abs_sync_error_s",
        "within_1_sample", "within_4_samples",
        "regression_slope", "regression_intercept", "correlation_coef",
        "snr",
        "n_saccades", "avg_saccade_amplitude", "avg_saccade_duration_s",
        "warnings_errors",
    ]

    all_keys = union_keys(rows)
    front = ["file_path", *BIDS_ENTITIES]
    ordered = [k for k in front if k in all_keys]
    ordered += [k for k in preferred_metrics if k in all_keys and k not in ordered]
    # Append whatever remains
    ordered += [k for k in all_keys if k not in set(ordered)]

    return rows, ordered


def write_excel(rows: List[Dict[str, Any]], columns: List[str], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    # Reindex to requested column order (missing columns auto-filled)
    df = df.reindex(columns=columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"Wrote {len(df)} rows to: {out_path}")


def main() -> None:
    args = parse_args()
    derivs_root: Path = args.derivs_root.resolve()
    out_path: Path = args.out if args.out.is_absolute() else (derivs_root / args.out)

    files = rglob_metrics(derivs_root, args.pattern)
    if not files:
        print(f"No files matched '{args.pattern}' under: {derivs_root}")
        # Still create an empty file with headers for convenience
        write_excel([], ["file_path", *BIDS_ENTITIES], out_path)
        return

    rows, columns = build_rows(files, verbose=args.verbose)
    write_excel(rows, columns, out_path)


if __name__ == "__main__":
    main()
