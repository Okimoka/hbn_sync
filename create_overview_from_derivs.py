"""
Create an Excel overview from *_proc-eyelink_metrics.json files (LLM Code)

Usage:
  python create_overview_from_derivs.py \
    --derivs-root ./extractedDataset/derivs \
    --out sync_metrics_overview.xlsx
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import re

PATTERN = "*_proc-eyelink_metrics.json"

LEGEND = {
    "subject": "Anonymized identifier for the subject",
    "release": "Release number which this subject appeared in",
    "TODO": "TODO"
}


def main():
    p = argparse.ArgumentParser(description="Summarize metrics JSONs to a single .xlsx")
    p.add_argument("--derivs-root", type=Path, default=Path("./extractedDataset/derivs"),
                   help="Path to derivatives root (default: ./extractedDataset/derivs)")
    p.add_argument("--out", type=Path, default=Path("sync_metrics_overview.xlsx"),
                   help="Output Excel filename (default: sync_metrics_overview.xlsx). "
                        "If relative, it will be saved under --derivs-root.")
    args = p.parse_args()

    root = args.derivs_root.resolve()
    out_path = args.out if args.out.is_absolute() else (root / args.out)

    files = sorted(root.rglob(PATTERN))
    rows = []
    columns = []  # first-seen column order across all files
    taskname = None

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)  # assume numbers, strings, or NaN
        # preserve per-file key order and append any new keys globally in first-seen order
        if taskname is None and "task" in data:
            taskname = data["task"]
        for k in data.keys():
            if k not in columns:
                columns.append(k)
        rows.append(data)

    df = pd.DataFrame(rows)
    if columns:
        df = df.reindex(columns=columns)

    legend_df = pd.DataFrame({"column": list(LEGEND.keys()),
                              "description": list(LEGEND.values())})

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=taskname, index=False)
        legend_df.to_excel(xw, sheet_name="legend", index=False)

    print(f"Wrote {len(df)} rows, {len(df.columns)} columns -> {out_path} "
          f"(main sheet: '{taskname}', legend rows: {len(legend_df)})")

if __name__ == "__main__":
    main()

