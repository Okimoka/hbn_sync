#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, List, ClassVar
import argparse
import math
import os
import numpy as np
import pandas as pd

"""
Create a "score" column in the subject metrics overview.
This is done by testing each subject against various quality checks, and subtracting from
their score (max. 100) if they fail.
"""


Direction = Literal["below", "above"]

"""
Helper class to define a quality metric as a combination of
- formula
- over/under
- absolute value / percentile
E.g. the criterion

Criterion(
    metric="within_1_sample / shared_events",
    direction="below",
    threshold_percentile=10,
    name="bottom_10pct_within_1_sample",
)

matches all subjects that are in the bottom 10 percentile when ranking them by
"within_1_sample / shared_events" (ratio of events that are within 1 sample of 0 lag).

"""
@dataclass
class Criterion:
    _default_df: ClassVar[Optional[pd.DataFrame]] = None
    _default_subject_col: ClassVar[str] = "subject"

    metric: str
    direction: Direction
    threshold_percentile: Optional[float] = None  # in [0,100]
    threshold_value: Optional[float] = None
    inclusive: bool = True
    name: Optional[str] = None
    df: Optional[pd.DataFrame] = field(default=None, repr=False)
    subject_col: Optional[str] = "subject"

    _realized_percentile: Optional[float] = field(default=None, init=False, repr=False)
    _last_mask: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _last_affected: Optional[List] = field(default=None, init=False, repr=False)

    @classmethod
    def configure_defaults(cls, df: pd.DataFrame, subject_col: str = "subject") -> None:
        cls._default_df = df
        cls._default_subject_col = subject_col

    def __post_init__(self):
        if self.df is not None:
            self.set_data(self.df, self.subject_col or "subject")
        elif Criterion._default_df is not None:
            self.set_data(Criterion._default_df, self.subject_col or Criterion._default_subject_col)

    # ---------- Core computation ----------
    def _eval_metric(self, df: pd.DataFrame) -> pd.Series:
        env = {str(c): df[c] for c in df.columns}
        safe_globals = {"__builtins__": {}, "np": np, "math": math, "pd": pd, "abs": abs}
        try:
            if self.metric in env:
                s = env[self.metric]
            else:
                s = eval(self.metric, safe_globals, env)  # intentionally allowed
        except Exception as e:
            raise ValueError(f"Failed to evaluate metric expression '{self.metric}': {e}")
        return pd.to_numeric(s, errors="coerce")


    def _fit_thresholds(self, series: pd.Series) -> Tuple[float, float]:
        valid = series.dropna().to_numpy(dtype=float)
        if valid.size == 0:
            raise ValueError("Metric evaluates to all-NaN. Cannot compute thresholds.")
        if self.threshold_value is None and self.threshold_percentile is None:
            raise ValueError("Provide either threshold_value or threshold_percentile.")
        if self.threshold_percentile is not None:
            p = float(self.threshold_percentile)
            if not np.isfinite(p) or p < 0.0 or p > 100.0:
                raise ValueError("threshold_percentile must be in [0, 100].")
            try:
                method = 'lower' if self.direction == 'below' else 'higher'
                t_val = float(np.nanpercentile(valid, p, method=method))
            except TypeError:
                interp = 'lower' if self.direction == 'below' else 'higher'
                t_val = float(np.nanpercentile(valid, p, interpolation=interp))
            self.threshold_value = t_val
            realized = (valid <= t_val).mean()*100 if self.direction=='below' else (valid >= t_val).mean()*100
            self._realized_percentile = float(realized)
        else:
            t_val = float(self.threshold_value)
            self._realized_percentile = float(
                (valid <= t_val).mean()*100 if self.direction=='below' else (valid >= t_val).mean()*100
            )
            self.threshold_percentile = self._realized_percentile
        return float(self.threshold_value), float(self.threshold_percentile)

    def _ensure_fitted(self):
        if self.df is None:
            return
        need = self.threshold_value is None or self._realized_percentile is None
        if need:
            s = self._eval_metric(self.df)
            self._fit_thresholds(s)

    # ---------- Public API ----------
    def set_data(self, df: pd.DataFrame, subject_col: Optional[str] = "subject") -> "Criterion":
        self.df = df
        self.subject_col = subject_col or "subject"
        if self.subject_col not in df.columns:
            raise KeyError(f"Required subject column '{self.subject_col}' not found in DataFrame.")
        self._clear_cache()
        return self

    def _clear_cache(self):
        self._last_mask = None
        self._last_affected = None

    def mask(self, df: Optional[pd.DataFrame] = None) -> pd.Series:
        the_df = df if df is not None else self.df
        if the_df is None:
            raise ValueError("No DataFrame bound. Pass df=... or call set_data(df).")
        s = self._eval_metric(the_df)
        t_val, _ = self._fit_thresholds(s)
        if self.direction == "below":
            m = s <= t_val if self.inclusive else s < t_val
        elif self.direction == "above":
            m = s >= t_val if self.inclusive else s > t_val
        else:
            raise ValueError("direction must be 'below' or 'above'")
        return m.fillna(False)

    def affected_subjects(self, df: Optional[pd.DataFrame] = None, subject_col: Optional[str] = None) -> List:
        the_df = df if df is not None else self.df
        if the_df is None:
            raise ValueError("No DataFrame bound. Pass df=... or call set_data(df).")
        col = subject_col or self.subject_col or "subject"
        if col not in the_df.columns:
            raise KeyError(f"Required subject column '{col}' not found in DataFrame.")
        m = self.mask(the_df)
        return the_df.loc[m, col].astype(str).tolist()

    @property
    def mask_series(self) -> pd.Series:
        if self._last_mask is None:
            self._last_mask = self.mask()
        return self._last_mask

    @property
    def affected(self) -> List:
        if self._last_affected is None:
            self._last_affected = self.affected_subjects()
        return self._last_affected

    @property
    def affected_count(self) -> int:
        return int(self.mask_series.sum())

    def describe(self) -> str:
        self._ensure_fitted()
        label = self.name or self.metric
        req = "n/a" if self.threshold_percentile is None else f"{self.threshold_percentile:.6f}"
        real = "n/a" if self._realized_percentile is None else f"{self._realized_percentile:.6f}"
        v = "n/a" if self.threshold_value is None else f"{self.threshold_value:.6g}"
        return f"[{label}] {self.direction} threshold. value={v}, requested_pct={req}, realized_pct={real}"

    def report(self, preview: int = 25) -> str:
        if self.df is None:
            raise ValueError("No DataFrame bound. Call set_data(df, 'subject') first.")
        _ = self.mask_series
        parts = [
            self.describe(),
            f"Affected rows: {self.affected_count}",
        ]
        if self.affected:
            tail = " ..." if len(self.affected) > preview else ""
            parts.append(f"First affected subjects: {self.affected[:preview]}{tail}")
        return "\n".join(parts)

# ---------- helpers ----------
def _parse_sheet_arg(sheet_arg: Optional[str]):
    if sheet_arg is None:
        return 0
    try:
        return int(sheet_arg)
    except (TypeError, ValueError):
        return sheet_arg

def _scored_filename(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_scored{ext or '.xlsx'}"

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Score subjects by EEG + ET quality criteria.")
    ap.add_argument("--xlsx", type=str, default="sync_metrics_overview.xlsx", help="Path to input .xlsx")
    ap.add_argument("--sheet", type=str, default=None, help="Excel sheet name or 0-based index")
    ap.add_argument("--output", type=str, default=None, help="Output .xlsx (default: <input>_scored.xlsx)")
    args = ap.parse_args()

    sheet_name = _parse_sheet_arg(args.sheet)
    df = pd.read_excel(args.xlsx, sheet_name=sheet_name)
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        df = df[first_key]

    if "subject" not in df.columns:
        raise KeyError("Required subject column 'subject' not found in the Excel sheet.")

    Criterion.configure_defaults(df, subject_col="subject")


    # ---------- criteria ----------
    # If either of these matches, subtract 10 points
    w1s = Criterion(
        metric="within_1_sample / shared_events",
        direction="below",
        threshold_percentile=10,
        name="bottom_10pct_within_1_sample",
    )
    w4s = Criterion(
        metric="within_4_samples / shared_events",
        direction="below",
        threshold_percentile=3,
        name="bottom_3pct_within_4_samples",
    )
    group_any_minus10 = [w1s, w4s]

    
    # For each of these checks, subtract 10 points if it applies
    # TODO rationalize each choice

    snr = Criterion(metric="snr", direction="below", threshold_value=4, name="snr_below_4")

    et_hz_below_58 = Criterion(metric="et_sampling_rate_hz", direction="below", threshold_value=58, name="et_hz_below_58")

    avg_sacc_amp_gt_20 = Criterion(metric="avg_saccade_amplitude", direction="above", threshold_value=20, name="avg_sacc_amp_gt_20")

    eeg_trim_ratio_gt_0_6 = Criterion(
        metric="eeg_samples_trimmed / eeg_samples",
        direction="above",
        threshold_value=0.6,
        name="eeg_trim_ratio_gt_0.6",
    )

    avg_blink_ms_gt_2000 = Criterion(metric="avg_blink_duration_ms", direction="above", threshold_value=2000, name="avg_blink_ms_gt_2000")

    avg_sacc_ms_gt_150 = Criterion(metric="avg_saccade_duration_ms", direction="above", threshold_value=150, name="avg_sacc_ms_gt_150")

    mean_abs_sync_ms_gt_1_75 = Criterion(metric="mean_abs_sync_error_ms", direction="above", threshold_value=1.75, name="mean_abs_sync_ms_gt_1.75")

    shared_events_lt_10 = Criterion(metric="shared_events", direction="below", threshold_value=10, name="shared_events_lt_10")

    xcorr_cos_sim_lt_0_85 = Criterion(metric="xcorr_cosine_similarity", direction="below", threshold_value=0.85, name="xcorr_cos_sim_lt_0.85")

    xcorr_n_peaks_spike_gt_5 = Criterion(metric="xcorr_n_peaks_spike", direction="above", threshold_value=5, name="xcorr_n_peaks_spike_gt_5")

    xcorr_peak_bottom_10pct = Criterion(metric="xcorr_peak", direction="below", threshold_percentile=10, name="xcorr_peak_bottom_10pct")

    abs_peak_idx_over_hz_gt_50 = Criterion(
        metric="abs(xcorr_peak_idx) / et_sampling_rate_hz",
        direction="above",
        threshold_value=50,
        name="abs_peak_idx_over_hz_gt_50",
    )

    minus10_criteria = [
        snr,
        et_hz_below_58,
        avg_sacc_amp_gt_20,
        eeg_trim_ratio_gt_0_6,
        avg_blink_ms_gt_2000,
        avg_sacc_ms_gt_150,
        mean_abs_sync_ms_gt_1_75,
        shared_events_lt_10,
        xcorr_cos_sim_lt_0_85,
        xcorr_n_peaks_spike_gt_5,
        xcorr_peak_bottom_10pct,
        abs_peak_idx_over_hz_gt_50,
    ]


    # If the xcorr at 0 lag is negative, subtract 100 points
    xcorr_zero_neg = Criterion(metric="xcorr_zero", direction="below", threshold_value=0, inclusive=False, name="xcorr_zero_lt_0")

    # Another metric that should cause full point subtraction, is an "availability" of "unavailable". These have been pre-filtered in this case
    
    # ---------- scoring ----------
    # Vectorized penalties
    p_any_group = pd.Series(False, index=df.index)
    for c in group_any_minus10:
        p_any_group = p_any_group | c.mask_series
    penalty_minus10 = p_any_group.astype(int) * 10

    for c in minus10_criteria:
        penalty_minus10 = penalty_minus10 + c.mask_series.astype(int) * 10

    penalty_minus100 = xcorr_zero_neg.mask_series.astype(int) * 100

    total_penalty = penalty_minus10 + penalty_minus100
    scores = 100 - total_penalty

    # Add to DataFrame
    out_df = df.copy()
    out_df["score"] = scores.astype(int)

    # Write output
    out_path = args.output or _scored_filename(args.xlsx)
    out_df.to_excel(out_path, index=False)

    print(f"Wrote scores to: {out_path}")
    print(f"Mean score: {out_df['score'].mean():.2f}")
    print(f"Min/Max score: {out_df['score'].min()} / {out_df['score'].max()}")

if __name__ == "__main__":
    main()
