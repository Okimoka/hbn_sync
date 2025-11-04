"""
Simple histogram script (LLM Code)
Intended as a helper to determine good thresholds to use for filter_subjects.py
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Hardcoded settings ----------
EXCEL_PATH = "sync_metrics_overview.xlsx"
SUBJECT_COL = "subject"

METRIC_EXPR = "eeg_samples_trimmed / eeg_samples"

# Number of histogram bins
BINS = 50
# ---------------------------------------

def compute_bins(values: np.ndarray, nbins: int) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        # All values identical — make a tiny range so the histogram renders
        eps = 0.5 if vmin == 0 else abs(vmin) * 0.01
        return np.array([vmin - eps, vmax + eps], dtype=float)
    return np.linspace(vmin, vmax, nbins + 1)

def compute_metric_series(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    Returns a numeric Series for either a column name or a simple expression.
    Unknown columns or invalid expressions cause a ValueError.
    """
    if expr in df.columns:
        s = pd.to_numeric(df[expr], errors="coerce")
    else:
        # Evaluate expression with columns as variables
        try:
            s = df.eval(expr)
        except Exception as e:
            raise ValueError(f"Cannot evaluate expression '{expr}': {e}")
        # Coerce to numeric, drop infinities from divisions by zero
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

    return s

def main():
    sheets = pd.read_excel(EXCEL_PATH, sheet_name=None)

    for sheet_name, df in sheets.items():
        # Require subject_id present
        if SUBJECT_COL not in df.columns:
            print(f"[SKIP] '{sheet_name}': missing '{SUBJECT_COL}'.")
            continue

        try:
            metric_series = compute_metric_series(df, METRIC_EXPR)
        except ValueError as e:
            print(f"[SKIP] '{sheet_name}': {e}")
            continue

        # Keep rows with subject_id and valid metric
        subset = df.loc[df[SUBJECT_COL].notna()].copy()
        values = metric_series.loc[subset.index].dropna()

        if values.empty:
            print(f"[SKIP] '{sheet_name}': no valid values for '{METRIC_EXPR}'.")
            continue

        edges = compute_bins(values.to_numpy(), BINS)

        # One figure per sheet
        plt.figure()
        plt.hist(values.to_numpy(), bins=edges)
        AS_PERCENT = False
        unit = " (%)" if AS_PERCENT else ""
        plt.title(f"{sheet_name} — {METRIC_EXPR}{unit}")
        plt.xlabel(f"{METRIC_EXPR}{unit}")
        plt.ylabel("Count")
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
