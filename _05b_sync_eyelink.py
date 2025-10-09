from types import SimpleNamespace
from datetime import datetime, timezone
from collections import Counter
import mne
import os.path
import re
import json
import csv
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.signal import correlate, find_peaks
from scipy.integrate import simpson
from ..._config_utils import (
    _bids_kwargs,
    get_eeg_reference,
    get_runs,
    get_sessions,
    get_subjects,
)
from ..._import_data import annotations_to_events, make_epochs
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._reject import _get_reject
from ..._report import _open_report
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs


"""
Captured metrics include

Meta:
- subject
- release
- task
- rating in participants.tsv

EEG:
- sampling rate
- sample count
- duration
- samples trimmed
- number of nan values
- channel count

ET:
- sampling rate
- sample count
- duration
- samples trimmed
- number of nan values
- channel count

- has "L POR X [px]", "R POR X [px]"?
- format of et (tsv, txt?)
- has _events?
- does _events match _samples events list?

- n_saccades
- avg_saccade_amplitude
- avg_saccade_duration_s
- avg_fixation_duration_ms

- n_blinks
- avg_blink_duration_ms


SYNC
- shared event count
- mean_abs_sync_error_ms
- within_1_sample
- within_4_samples
- regression slope
- regression_intercept
- correlation_coef

TODO
- snr of eeg
- out of range et samples? (i think eye-eeg did this with deg_per_px, viewing distance)
- (avg fixation positions)


"""


#Function that resembles a "perfect" cross-correlation plot
def xcorr_template_curve(x, k: float = 8.0, a: float = 5.0) -> np.ndarray:
    #x = np.asarray(x, dtype=float)
    #z = k * x
    #y = np.empty_like(z)
    #zero = (z == 0)
    #y[zero] = 1.0
    #nz = ~zero
    #y[nz] = np.sin(z[nz]) / z[nz]
    #return y * np.exp(-((z / a) ** 2))

    x = np.asarray(x, dtype=float)
    return np.maximum(0.0, 1.0 - np.abs(x))





def normalized_xcorr(s1, s2, mode="full"):
    """Demean both signals, compute cross-correlation, and L2-normalize.
    Returns zeros if either signal is flat."""
    s1 = np.asarray(s1, float)
    s2 = np.asarray(s2, float)
    s1 = s1 - np.mean(s1)
    s2 = s2 - np.mean(s2)

    corr = correlate(s1, s2, mode=mode)
    denom = np.linalg.norm(s1) * np.linalg.norm(s2)
    if denom > 0:
        return corr / denom
    return np.zeros_like(corr)



def _samples_events_matches(ann_on, ann_ds, inline_msgs, t0, *, strict=True):

    # From annotations: keep only those whose description contains "Message"
    ann_pairs = [
        (int(round(on * 1e6 + t0)), ds.strip())
        for on, ds in zip(ann_on, ann_ds)
        if "Message" in ds
    ]

    return inline_msgs == ann_pairs


#TODO peak-to-peak / std for snr
"""
def _compute_snr(...):
    return None
"""


def _read_raw_iview(sample_fname: str, event_fname: str | None = None):

    with open(sample_fname, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # One of the first lines will be "## Sample Rate: 60"
    sfreq_idx = next(i for i, ln in enumerate(lines) if ln.startswith("##") and "Sample Rate" in ln)
    sfreq = float(lines[sfreq_idx].split(':')[1].strip())

    #First column header is always "Time"
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Time"))
    header = lines[header_idx].rstrip("\n").split("\t")
    time_idx = header.index("Time")
    type_idx = header.index("Type")

    num_idx = [i for i in range(len(header)) if i not in (type_idx,)] # type is the only non-numeric column
    ch_names = [header[i] for i in num_idx if i not in (time_idx,)] # time is not a channel
    data_cols = [[] for _ in num_idx]
    inline_msgs = []

    for ln in lines[header_idx + 1:]:
        parts = ln.rstrip("\n").split("\t")

        # inline MSG row, looks like "2655102177 MSG 1 # Message: 20"
        if len(parts) == 4:
            assert parts[type_idx] == "MSG"
            inline_msgs.append((float(parts[time_idx]), parts[-1].strip()))
            continue
        for j, ci in enumerate(num_idx):
            v = parts[ci]
            data_cols[j].append(np.nan if v == "" else float(v))

    # often, the header has more entries than the rows have values, these will be all nan
    nan_cols = [np.all(np.isnan(col)) for col in data_cols]
    data_cols = [col for col, is_nan in zip(data_cols, nan_cols) if not is_nan]

    # drop these channels for the ch_names object as well
    ch_names = [nm for nm, is_nan in zip(ch_names, nan_cols[1:]) if not is_nan]

    times = np.asarray(data_cols[0], float)
    data = np.vstack([np.asarray(c, float) for c in data_cols[1:]])

    sfreq_est = 1.0 / (np.median(np.diff(times)) / 1e6)  # Time is in µs
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["misc"] * len(ch_names))
    raw_et = mne.io.RawArray(data, info)

    #Data reading done. Now read Annotations from _Events.txt (if available)

    ann_on, ann_du, ann_ds, ann_ex = [], [], [], [] #onset, duration, description, extras
    et_eventfile_matches = False
    et_unknown_events = 0
    t0 = times[0]

    #TODO Table headers are listed at the start of the file - use these instead of hardcoded len(parts) check

    if event_fname and os.path.exists(event_fname):
        with open(event_fname, "r", encoding="utf-8", errors="ignore") as ef:
            for parts in csv.reader(ef, delimiter="\t"):
                #empty lines
                if(not parts):
                    continue
                kind = parts[0]
                #Looks like "UserEvent 1 5 2547585001 # Message: 14"
                if len(parts) == 5 and kind.startswith("UserEvent"):
                    start = float(parts[3])
                    # "The starting time of annotations in seconds after orig_time"
                    # "Where orig_time is determined from beginning of raw data acquisition"
                    ann_on.append((start - t0) / 1e6)
                    ann_du.append(0.0) # Duration column is only used for saccades/fixations
                    ann_ds.append(parts[4].strip()) # Message: 14
                    ann_ex.append(None) # No extra data for UserEvents
                #Looks like (no commas) "Fixation L, 1, 1, 2524661981, 2524811929, 149948, 1077.17, 697.01, 29, 17, -1, 12.61, 12.61
                elif len(parts) >= 5 and (kind.startswith("Fixation") or kind.startswith("Saccade") or kind.startswith("Blink")):
                    start, end = float(parts[3]), float(parts[4])
                    ann_on.append((start - t0) / 1e6)
                    # Alternatively, duration could be pulled from parts[5] directly
                    ann_du.append((end - start) / 1e6)
                    ann_ds.append(kind)

                    #TODO use table header here as well
                    extras_dict = {f"extra_{k}": float(val) for k, val in enumerate(parts[5:])}
                    ann_ex.append(extras_dict or None)

                                  
                else:
                    et_unknown_events += 1
            
            #check whether inline_msgs matches the events from this file
            et_eventfile_matches = _samples_events_matches(ann_on, ann_ds, inline_msgs, t0)
    else:
        logger.info(**gen_log_kwargs(message=f"SMI _Events.txt not found, using Events from _Samples.txt"))
        # Fallback: inline MSG rows as zero-duration annotations
        for t_us, desc in inline_msgs:
            ann_on.append((t_us - t0) / 1e6)
            ann_du.append(0.0)
            ann_ds.append(desc)
            ann_ex.append(None) # No extra data for inline messages
    if ann_on:
        raw_et.set_annotations(mne.Annotations(onset=ann_on, duration=ann_du, description=ann_ds, extras=ann_ex))

    return raw_et, et_eventfile_matches, et_unknown_events



def _check_HEOG_ET_vars(cfg):
    # helper function for sorting out heog and et channels
    bipolar = False
    if isinstance(cfg.sync_heog_ch, tuple):
        heog_ch = "bi_HEOG"
        bipolar = True
    else:
        heog_ch = cfg.sync_heog_ch
    
    if isinstance(cfg.sync_et_ch, tuple):
        et_ch = list(cfg.sync_et_ch)
    else:
        et_ch = [cfg.sync_et_ch]
    
    return heog_ch, et_ch, bipolar

def _mark_calibration_as_bad(raw, cfg):
    # marks recalibration beginnings and ends as one bad segment
    cur_idx = None
    cur_start_time = 0.
    last_status = None
    for annot in raw.annotations:
        calib_match = re.match(cfg.sync_calibration_string, annot["description"])
        if not calib_match: continue
        calib_status, calib_idx = calib_match.group(1), calib_match.group(2)
        if calib_idx  == cur_idx and calib_status == "end":
            duration = annot["onset"] - cur_start_time
            raw.annotations.append(cur_start_time, duration, f"BAD_Recalibrate {calib_idx}")
            cur_idx, cur_start_time = None, 0.
        elif calib_status == "start" and cur_idx is None:
            cur_idx = calib_idx
            cur_start_time = annot["onset"]
        elif calib_status == last_status:
            logger.info(**gen_log_kwargs(message=f"Encountered apparent duplicate calibration event ({calib_status}, {calib_idx}) - skipping"))
        elif calib_status == "start" and cur_idx is not None:
            raise ValueError(f"Annotation {annot["description"]} could not be assigned membership"
                             f"")
        last_status = calib_status
        
    return raw
        

def get_input_fnames_sync_eyelink(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> dict:
    
    # Get from config file whether `task` is specified in the et file name
    if cfg.et_has_task == True:
        et_task = cfg.task
    else:
        et_task = None

    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
        extension=".fif",
    )

    et_asc_bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=et_task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        datatype="beh",
        root=cfg.bids_root,
        suffix="et",
        check=False,
        extension=".asc",
    )

    et_edf_bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=et_task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        datatype="beh",
        root=cfg.bids_root,
        suffix="et",
        check=False,
        extension=".edf",
    )

    et_txt_bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=et_task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        datatype="beh",
        root=cfg.bids_root,
        suffix="et",
        check=False,
        extension=".txt",
    )

    in_files = dict()
    for run in cfg.runs:
        key = f"raw_run-{run}"
        in_files[key] = bids_basename.copy().update(
            run=run, processing=cfg.processing, suffix="raw"
        )
        _update_for_splits(in_files, key, single=True)

        et_bids_basename_temp = et_asc_bids_basename.copy()

        if cfg.et_has_run:
            et_bids_basename_temp.update(run=run)

        # _update_for_splits(in_files, key, single=True) # TODO: Find out if we need to add this or not

        if not os.path.isfile(et_bids_basename_temp):
            logger.info(**gen_log_kwargs(message=f"Couldn't find {et_bids_basename_temp} file. If edf file exists, edf2asc will be called."))

            et_bids_basename_temp = et_edf_bids_basename.copy()

            if cfg.et_has_run:
                et_bids_basename_temp.update(run=run)

            # _update_for_splits(in_files, key, single=True) # TODO: Find out if we need to add this or not

            if not os.path.isfile(et_bids_basename_temp):

                # Try txt
                et_bids_basename_temp = et_txt_bids_basename.copy()
                if cfg.et_has_run:
                    et_bids_basename_temp.update(run=run)
                if not os.path.isfile(et_bids_basename_temp):
                    logger.error(**gen_log_kwargs(message=f"Also didn't find {et_bids_basename_temp} file, one of .asc, .edf or .txt needs to exist for ET sync."))
                    raise FileNotFoundError(f"For run {run}, could neither find .asc, .edf nor .txt eye-tracking file. Please double-check the file names.")

        key = f"et_run-{run}"
        in_files[key] = et_bids_basename_temp
  
    return in_files



@failsafe_run(
    get_input_fnames=get_input_fnames_sync_eyelink,
)
def sync_eyelink(
 *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: dict,
) -> dict:
    
    """Run Sync for Eyelink."""
    import matplotlib.pyplot as plt


    raw_fnames = [in_files.pop(f"raw_run-{run}") for run in cfg.runs]
    et_fnames = [in_files.pop(f"et_run-{run}") for run in cfg.runs]
    
    logger.info(**gen_log_kwargs(message=f"Found the following eye-tracking files: {et_fnames}"))
    out_files = dict()
    bids_basename = raw_fnames[0].copy().update(processing=None, split=None, run=None)
    out_files["eyelink"] = bids_basename.copy().update(processing="eyelink", suffix="raw")
    del bids_basename



    
    participants_info = dict(
        release_number=np.nan,
        availability=np.nan
    )

    if os.path.isfile(cfg.bids_root / "participants.tsv"):
        with (cfg.bids_root / "participants.tsv").open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if (row.get('participant_id') or '').strip() == f"sub-{subject}":

                    release_val = (row.get('release_number') or '').strip()
                    if release_val:
                        participants_info['release_number'] = release_val

                    # Update the value at column named by cfg.task if present/non-empty
                    task_val = (row.get(cfg.task) or '').strip()
                    if task_val:
                        participants_info["availability"] = task_val




    metrics = dict(
        subject=subject,
        release=participants_info["release_number"],
        task=cfg.task,
        availability=participants_info["availability"],

        eeg_samples=np.nan,
        et_samples=np.nan,
        eeg_sampling_rate_hz=np.nan,
        et_sampling_rate_hz=np.nan,
        eeg_samples_trimmed=np.nan,
        et_samples_trimmed=np.nan,
        eeg_nan_values=np.nan,
        et_nan_values=np.nan,
        eeg_channel_cnt=np.nan,
        et_channel_cnt=np.nan,

        et_has_PORX=np.nan,
        et_data_format=np.nan,
        et_has_eventfile=False,
        et_eventfile_matches=np.nan,
        et_unknown_events=np.nan,

        shared_events=np.nan,
        mean_abs_sync_error_ms=np.nan,
        within_1_sample=np.nan,
        within_4_samples=np.nan,
        regression_slope=np.nan,
        regression_intercept=np.nan,
        correlation_coef_zero=np.nan,
        correlation_coef_peak=np.nan,
        correlation_coef_peak_idx=np.nan,
        correlation_coef_second_peak=np.nan,
        correlation_coef_second_peak_idx=np.nan,
        snr=np.nan,

        xcorr_cosine_similarity=np.nan,
        xcorr_rmse=np.nan,
        xcorr_dinf=np.nan,
        xcorr_scale=np.nan,
        xcorr_kl=np.nan,

        n_saccades=np.nan,
        avg_saccade_amplitude=np.nan,
        avg_saccade_duration_ms=np.nan,
        avg_fixation_duration_ms=np.nan,

        n_blinks=np.nan,
        avg_blink_duration_ms=np.nan

    )



    
    for idx, (run, raw_fname,et_fname) in enumerate(zip(cfg.runs, raw_fnames,et_fnames)):
        msg = f"Syncing Eyelink ({et_fname.basename}) and EEG data ({raw_fname.basename})."
        logger.info(**gen_log_kwargs(message=msg))
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

        et_format = et_fname.extension
        metrics["et_data_format"] = et_format

        if et_format == '.edf':
            logger.info(**gen_log_kwargs(message=f"Converting {et_fname} file to `.asc` using edf2asc."))
            import subprocess
            subprocess.run(["edf2asc", et_fname]) # TODO: Still needs to be tested
            et_fname.update(extension='.asc')
            raw_et = mne.io.read_raw_eyelink(et_fname, find_overlaps=True)
        elif et_format == '.asc':
            raw_et = mne.io.read_raw_eyelink(et_fname, find_overlaps=True)
        elif et_format == '.txt':
            # Attempt to find corresponding events file (if it exists)
            event_fname = None
            base_root = os.path.splitext(str(et_fname))[0]
            logger.info(**gen_log_kwargs(message=f"Looking for {str(base_root + '_Events.txt')} "))
            if os.path.isfile(base_root + '_Events.txt'):
                event_fname = base_root + '_Events.txt'
                metrics["et_has_eventfile"] = True

            raw_et, et_eventfile_matches, et_unknown_events = _read_raw_iview(str(et_fname), event_fname)
            metrics["et_eventfile_matches"] = et_eventfile_matches
            metrics["et_unknown_events"] = et_unknown_events
        else:
            
            raise AssertionError("ET file is neither an `.asc`, `.edf`, nor `.txt`.")


        metrics["eeg_samples"] = raw.n_times
        metrics["et_samples"] = raw_et.n_times
        metrics["eeg_sampling_rate_hz"] = raw.info["sfreq"]
        metrics["et_sampling_rate_hz"] = raw_et.info["sfreq"]
        metrics["eeg_channel_cnt"] = len(raw.ch_names)
        #TODO maybe already cancel here if channel count != 43?
        #Will have to adjust per release
        metrics["et_channel_cnt"] = len(raw_et.ch_names)
        metrics["et_has_PORX"] = "L POR X [px]" in raw_et.ch_names and "R POR X [px]" in raw_et.ch_names


        # If the user did not specify a regular expression for the eye-tracking sync events, it is assumed that it's
        # identical to the regex for the EEG sync events
        if not cfg.sync_eventtype_regex_et:
            cfg.sync_eventtype_regex_et = cfg.sync_eventtype_regex
        
        et_sync_times = [annotation["onset"] for annotation in raw_et.annotations if re.search(cfg.sync_eventtype_regex_et,annotation["description"])]
        sync_times    = [annotation["onset"] for annotation in raw.annotations    if re.search(cfg.sync_eventtype_regex,   annotation["description"])]
        
        assert len(et_sync_times) == len(sync_times),f"Detected eyetracking and EEG sync events were not of equal size ({len(et_sync_times)} vs {len(sync_times)}). Adjust your regular expressions via 'sync_eventtype_regex_et' and 'sync_eventtype_regex' accordingly"
        #logger.info(**gen_log_kwargs(message=f"{et_sync_times}"))
        #logger.info(**gen_log_kwargs(message=f"{sync_times}"))

        _num_nans_this = 0
        # Check whether the eye-tracking data contains nan values. If yes replace them with zeros.
        if np.isnan(raw_et.get_data()).any():

            # Set all nan values in the eye-tracking data to 0 (to make resampling possible)
            # TODO: Decide whether this is a good approach or whether interpolation (e.g. of blinks) is useful
            # TODO: Decide about setting the values (e.g. for blinks) back to nan after synchronising the signals
            _num_nans_this = int(np.isnan(raw_et.get_data()).sum())
            np.nan_to_num(raw_et._data, copy=False, nan=0.0)
            logger.info(**gen_log_kwargs(message=f"The eye-tracking data contained nan values. They were replaced with zeros."))

        metrics["et_nan_values"] = _num_nans_this
        metrics["eeg_nan_values"] = int(np.isnan(raw._data).sum())

        et_pre_n, et_pre_f   = raw_et.n_times, float(raw_et.info["sfreq"])
        eeg_pre_n, eeg_pre_f = raw.n_times,    float(raw.info["sfreq"])

        # Align the data
        mne.preprocessing.realign_raw(raw, raw_et, sync_times, et_sync_times)

        metrics["et_samples_trimmed"]  = max(0, int(round(et_pre_n  - raw_et.n_times * (et_pre_f  / float(raw_et.info["sfreq"])))))
        metrics["eeg_samples_trimmed"] = max(0, int(round(eeg_pre_n - raw.n_times    * (eeg_pre_f / float(raw.info["sfreq"])))))



        # Add ET data to EEG
        raw.add_channels([raw_et], force_update_info=True)
        raw._raw_extras.append(raw_et._raw_extras)

        # Also add ET annotations to EEG
        # first mark et sync event descriptions so we can differentiate them later
        # prevent np fixed-width strings truncation when prefixing with ET_
        raw_et.annotations.description = raw_et.annotations.description.astype(object)
        for idx, desc in enumerate(raw_et.annotations.description):
            if re.search(cfg.sync_eventtype_regex_et, desc):
                raw_et.annotations.description[idx] =  "ET_" + desc


        comb = mne.annotations._combine_annotations(
            raw.annotations,
            raw_et.annotations,
            0,
            raw.first_samp,
            raw.first_samp,
            raw.info["sfreq"],
        )
        
        # When raw._first_time and raw_et._first_time are mismatched, the last events are cut off
        # Identify which descriptions came from the ET stream
        is_et = np.array([d in set(raw_et.annotations.description) for d in comb.description], dtype=bool)
        # Shift EEG annotations back; leave ET as-is
        comb.onset[~is_et] -= float(raw.first_samp) / float(raw.info["sfreq"])

        raw.set_annotations(mne.Annotations(onset=comb.onset, duration=comb.duration, description=comb.description, orig_time=None))


        msg = f"Saving synced data to disk."
        logger.info(**gen_log_kwargs(message=msg))
        raw.save(
            out_files["eyelink"],
            overwrite=True,
            split_naming="bids", # TODO: Find out if we need to add this or not
            split_size=cfg._raw_split_size, # ???
        )
        # no idea what the split stuff is...
        _update_for_splits(out_files, "eyelink") # TODO: Find out if we need to add this or not


    # Add to report
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 19.2))
    msg = f"Adding figure to report."
    logger.info(**gen_log_kwargs(message=msg))
    tags = ("sync", "eyelink")
    title = "Synchronize Eyelink"
    caption = (
           f"The `realign_raw` function from MNE was used to align an Eyelink `asc` file to the M/EEG file."
           f"The Eyelink-data was added as annotations and appended as new channels."
        )
    if cfg.sync_heog_ch is None or cfg.sync_et_ch is None:
        # we need both an HEOG channel and ET channel specified to do cross-correlation
        msg = f"HEOG and/or ET channel not specified; cannot produce cross-correlation for report."
        logger.info(**gen_log_kwargs(message=msg))
        caption += "\nHEOG and/or eye tracking channels were not specified and no cross-correlation was performed."
        axes[0,0].text(0.5, 0.5, 'HEOG/ET cross-correlation unavailable', fontsize=34,
                       horizontalalignment='center', verticalalignment='center', transform=axes[0,0].transAxes)
        axes[0,0].axis("off")
    else:
        # return _prep_out_files(exec_params=exec_params, out_files=out_files)
        # calculate cross correlation of HEOG with ET
        heog_ch, et_ch, bipolar = _check_HEOG_ET_vars(cfg)
        if bipolar:
            # create bipolar HEOG
            raw = mne.set_bipolar_reference(raw, *cfg.sync_heog_ch, ch_name=heog_ch, drop_refs=False)
        raw.filter(l_freq=cfg.sync_heog_highpass, h_freq=cfg.sync_heog_lowpass, picks=heog_ch) # get rid of drift and high freq noise
        _mark_calibration_as_bad(raw, cfg)
        # extract HEOG and ET as arrays
        heog_array = raw.get_data(picks=[heog_ch], reject_by_annotation="omit")
        et_array = raw.get_data(picks=et_ch, reject_by_annotation="omit")
        if len(et_array) > 1:
            et_array = et_array.mean(axis=0, keepdims=True)

        

        heog_signal = heog_array[0]
        et_signal = et_array.mean(axis=0) if len(et_array) > 1 else et_array[0]

        # r at zero-lag
        r0 = float(np.corrcoef(heog_signal, et_signal)[0, 1])
        metrics["correlation_coef_zero"] = r0

        # cross corr peak
        ncc = normalized_xcorr(heog_signal, et_signal, mode="full")
        metrics["correlation_coef_peak"] = float(np.max(np.abs(ncc))) if ncc.size else 0.0
        metrics["correlation_coef_peak_idx"] = int(np.argmax(np.abs(ncc)) - (len(et_signal) - 1)) if ncc.size else 0


        # second highest peak within +-3000 samples of 0
        if ncc.size:
            mid = len(et_signal) - 1
            lo, hi = max(0, mid - 3000), min(len(ncc), mid + 3000 + 1)
            win = ncc[lo:hi]
            pk, _ = find_peaks(win) # local maxima of ncc
            if pk.size >= 2:
                order = np.argsort(win[pk])[::-1] # sort peaks by height, desc
                j2 = lo + int(pk[order][1]) # 2nd-highest peak (global index)
                metrics["correlation_coef_second_peak"] = float(ncc[j2])
                metrics["correlation_coef_second_peak_idx"] = int(j2 - mid)  # lag in samples


        # normalized xcorr
        corr = normalized_xcorr(heog_signal, et_signal, mode="same")




        # Compare xcorr to template on +/-3000 samples (x in [-3, 3])
        mid = len(corr) // 2
        lo, hi = max(0, mid - 3000), min(len(corr), mid + 3000 + 1)
        samp_offsets = np.arange(lo - mid, hi - mid) # e.g., [-3000, ..., 3000]
        corr_win = corr[lo:hi]
        tmpl_win = xcorr_template_curve(samp_offsets / 1000.0)

        # Cosine similarity (scale-invariant)
        denom = np.linalg.norm(corr_win) * np.linalg.norm(tmpl_win)
        cos_sim = float(np.dot(corr_win, tmpl_win) / denom) if denom > 0 else np.nan

        # Best-fit scaling of template to minimize squared error
        s2 = float(np.dot(tmpl_win, tmpl_win))
        alpha = float(np.dot(corr_win, tmpl_win) / s2) if s2 > 0 else 0.0
        tmpl_fit = alpha * tmpl_win

        # Error metrics (after best-fit scaling)
        rmse = float(np.sqrt(np.mean((corr_win - tmpl_fit) ** 2)))
        dinf = float(np.max(np.abs(corr_win - tmpl_fit)))

        # Save metrics
        metrics["xcorr_cosine_similarity"] = cos_sim
        metrics["xcorr_rmse"] = rmse
        metrics["xcorr_dinf"] = dinf
        metrics["xcorr_scale"] = alpha

        metrics["xcorr_kl"] = 100



        # cross correlate them
        #corr = correlate(heog_array[0], et_array[0], mode="same") / heog_array.shape[1]
        # plot cross correlation
        # figure out how much we plot
        midpoint = len(corr) // 2
        plot_samps = (-cfg.sync_plot_samps, cfg.sync_plot_samps) if isinstance(cfg.sync_plot_samps, int) else cfg.sync_plot_samps
        if isinstance(plot_samps, tuple):
            x_range = np.arange(plot_samps[0], plot_samps[1])
            y_range = np.arange(midpoint+plot_samps[0], midpoint+plot_samps[1])
        else: # None
            y_range = np.arange(len(corr))
            x_range = y_range - midpoint
        # plot
        axes[0,0].plot(x_range, corr[y_range], color="black")

        # existing xcorr curve
        axes[0,0].plot(x_range, corr[y_range], color="black", label="XCorr (HEOG and ET)")

        #tmpl_unit_plot = xcorr_template_curve(x_range / 1000.0)
        #axes[0,0].plot(x_range, tmpl_unit_plot, linestyle="--", linewidth=2,
        #            label="Template function")






        """
        Awful LLM code
        Just to visualize what the template curve to compare with could look like
        Doesnt actually affect the metrics
        Optimally, only the central pyramid part of the curve should be compared with (TODO)


        # Scale, shift, and set width of the pyramid based on xcorr
        win = corr[y_range]
        default_half_width = 1000.0  # your current template width: zeros at ±1000 samples

        def _first_zero_right(arr, start_idx):
            seg = arr[start_idx:]
            if seg.size == 0:
                return None
            # exact zero first
            z = np.where(seg == 0)[0]
            if z.size:
                return start_idx + int(z[0])
            # sign change (between k-1 and k)
            prod = seg[:-1] * seg[1:]
            chg = np.where(prod < 0)[0]
            if chg.size:
                return start_idx + int(chg[0] + 1)
            return None

        def _first_zero_left(arr, start_idx):
            seg = arr[:start_idx + 1]
            if seg.size == 0:
                return None
            # any exact zero to the left (take the last one)
            z = np.where(seg == 0)[0]
            if z.size:
                return int(z[-1])
            # sign change scanning from the right
            prod = seg[:-1] * seg[1:]
            chg = np.where(prod < 0)[0]
            if chg.size:
                return int(chg[-1] + 1)
            return None

        if win.size:
            # peak based on absolute value in the plotted window
            i_peak = int(np.argmax(np.abs(win)))
            peak_val = float(np.abs(win[i_peak]))      # height of pyramid
            peak_lag = int(x_range[i_peak])            # lag (in samples) where the peak occurs
            # shift only if the peak is within ±100 samples of 0
            shift = peak_lag if abs(peak_lag) <= 100 else 0

            # infer width from first zero-crossings on each side of the PEAK (use raw win, not abs)
            i0_r = _first_zero_right(win, i_peak)
            i0_l = _first_zero_left(win, i_peak)

            if (i0_l is not None) and (i0_r is not None):
                # distances in SAMPLES using x_range (not just index math)
                d_left  = abs(int(x_range[i_peak]) - int(x_range[i0_l]))
                d_right = abs(int(x_range[i0_r])   - int(x_range[i_peak]))
                half_width = float(0.5 * (d_left + d_right))
                # sanity
                if not np.isfinite(half_width) or half_width <= 0:
                    half_width = default_half_width
            else:
                half_width = default_half_width

            # Compute steepness angle in degrees with x normalized like your default (1000 samples == 1 unit).
            # angle = atan( peak / (half_width/1000) ). If angle < 45°, fall back to default width.
            #TODO fix this - reports wrong angles
            angle_deg = float(np.degrees(np.arctan2(peak_val, half_width / 1000.0)))
            #print("COMPUTED ANGLE")
            #print(angle_deg)
            #if angle_deg < 45.0:
            #    half_width = default_half_width
            #    angle_deg = float(np.degrees(np.arctan2(peak_val, 1.0)))  # document angle actually used

            # save steepness to metrics
            metrics["xcorr_template_steepness_deg"] = angle_deg
        else:
            peak_val = 1.0
            shift = 0
            half_width = default_half_width
            metrics["xcorr_template_steepness_deg"] = float(np.degrees(np.arctan2(peak_val, half_width / 1000.0)))

        # build and plot the adapted pyramid
        tmpl_plot = peak_val * xcorr_template_curve((x_range - shift) / float(half_width))
        axes[0,0].plot(x_range, tmpl_plot, linestyle="--", linewidth=2, label="Template function")

        """




        axes[0,0].axvline(linestyle="--", alpha=0.3)
        axes[0,0].set_title("Cross correlation HEOG and ET")
        axes[0,0].set_xlabel("Samples")
        axes[0,0].set_ylabel("X correlation")
        # calculate delay
        delay_idx = abs(corr).argmax() - midpoint
        delay_time = delay_idx * (raw.times[1] - raw.times[0])
        caption += f"\nThere was an estimated synchronisation delay of {delay_idx} samples ({delay_time:.3f} seconds.)"
    
    # regression between synced events
    # we assume here that these annotations are sequential pairs of the same event in raw and et. otherwise this will break
    raw_onsets = [annot["onset"] for annot in raw.annotations if re.match(cfg.sync_eventtype_regex, annot["description"])]
    et_onsets = [annot["onset"] for annot in raw.annotations if re.match("ET_"+cfg.sync_eventtype_regex_et, annot["description"])]
 
    if len(raw_onsets) != len(et_onsets):
        raise ValueError(f"Lengths of raw {len(raw_onsets)} and ET {len(et_onsets)} onsets do not match.")
    
    metrics["shared_events"] = len(raw_onsets)

    # regress and plot
    coef = np.polyfit(raw_onsets, et_onsets, 1)
    preds = np.poly1d(coef)(raw_onsets)
    resids = et_onsets - preds
    axes[0,1].plot(raw_onsets, et_onsets, "o", alpha=0.3, color="black")
    axes[0,1].plot(raw_onsets, preds, "--k")
    axes[0,1].set_title("Regression")
    axes[0,1].set_xlabel("Raw onsets (seconds)")
    axes[0,1].set_ylabel("ET onsets (seconds)")
    # residuals
    axes[1,0].plot(np.arange(len(resids)), resids, "o", alpha=0.3, color="black")
    axes[1,0].axhline(linestyle="--")
    axes[1,0].set_title("Residuals")
    axes[1,0].set_ylabel("Residual (seconds)")
    axes[1,0].set_xlabel("Samples")
    # histogram of distances between events in time
    #axes[1,1].hist(np.array(raw_onsets) - np.array(et_onsets), bins=11, range=(-5,5), color="black")

    # histogram of distances in samples (-5 to +5 samples offset)
    diff_samples = (np.array(raw_onsets) - np.array(et_onsets)) * float(raw.info["sfreq"])
    bin_edges = np.arange(-5.5, 5.6, 1.0)  # [-5.5, -4.5, ..., 4.5, 5.5]
    axes[1, 1].hist(diff_samples, bins=bin_edges, color="black")
    print("Number of shared events: " + str(len(diff_samples)))

    axes[1,1].set_title("Raw - ET event onset distances histogram")
    axes[1,1].set_xlabel("milliseconds")
    # this doesn't seem to help, though it should...
    fig.tight_layout()

    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        task=cfg.task,
    ) as report:
        caption = caption
        report.add_figure(
            fig=fig,
            title="Eyelink data",
            section=title,
            caption=caption,
            tags=tags[1],
            replace=True,
        )
        plt.close(fig)
        del caption

    

    metrics["mean_abs_sync_error_ms"] = (float(np.mean(np.abs(resids))) * 1000.0 if resids.size > 0 else np.nan)


    if len(raw_onsets):
        _diff_samples_abs = np.abs((np.array(raw_onsets) - np.array(et_onsets)) * float(raw.info["sfreq"]))
        metrics["within_1_sample"] = int(np.sum(_diff_samples_abs <= 1.0))
        metrics["within_4_samples"] = int(np.sum(_diff_samples_abs <= 4.0))

    # regression + correlation across all events
    if len(raw_onsets):
        _coef_all = np.polyfit(raw_onsets, et_onsets, 1)
        metrics["regression_slope"] = float(_coef_all[0])
        metrics["regression_intercept"] = float(_coef_all[1])


    #print(raw_et.annotations["description"])
    # Saccade stats
    if getattr(raw_et, "annotations", None) is not None:
        _is_saccade = np.array([str(desc).startswith("Saccade") for desc in raw_et.annotations.description], dtype=bool)
        _is_fixation = np.array([str(desc).startswith("Fixation") for desc in raw_et.annotations.description], dtype=bool)
        _is_blink = np.array([str(desc).startswith("Blink") for desc in raw_et.annotations.description], dtype=bool)
        metrics["n_saccades"] = int(_is_saccade.sum())
        metrics["n_blinks"] = int(_is_blink.sum())

        metrics["avg_saccade_duration_ms"] = float(np.mean(np.asarray(raw_et.annotations.duration, float)[_is_saccade]) * 1000.0)
        metrics["avg_fixation_duration_ms"] = float(np.mean(np.asarray(raw_et.annotations.duration, float)[_is_fixation]) * 1000.0)
        metrics["avg_blink_duration_ms"] = float(np.mean(np.asarray(raw_et.annotations.duration, float)[_is_blink]) * 1000.0)
        
        #metrics["avg_saccade_amplitude"] = float(np.mean(np.asarray(raw_et.annotations.extra_5, float)[_is_saccade]))

        anns = raw_et.annotations.to_data_frame()
        filtered = anns.loc[pd.Series(_is_saccade, index=anns.index)]
        metrics["avg_saccade_amplitude"] = filtered["extra_5"].mean()


    ###metrics["snr"] = _compute_snr(raw, power_line_freq=60.0)

    #n_saccades=np.nan,
    #avg_saccade_amplitude=np.nan,
    #avg_saccade_duration_ms=np.nan,
    #avg_fixation_duration_ms=np.nan,

    #n_blinks=np.nan,
    #avg_blink_duration_ms=np.nan


    metrics_bids = raw_fnames[0].copy().update(
        run=None, split=None,
        processing="eyelink",
        suffix="metrics",
        extension=".json",
    )

    metrics_bids.fpath.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(metrics_bids.fpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)


    #print("Causing crash now")
    #print(raw_et.annotations["description"])

    return _prep_out_files(exec_params=exec_params, out_files=out_files)



def get_config(
   *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
) -> SimpleNamespace:
    #logger.info(**gen_log_kwargs(message=f"config {config}"))

    cfg = SimpleNamespace(
        runs=get_runs(config=config, subject=subject),
        remove_blink_saccades   = config.remove_blink_saccades,
        et_has_run = config.et_has_run,
        et_has_task = config.et_has_task,
        sync_eventtype_regex    = config.sync_eventtype_regex,
        sync_eventtype_regex_et = config.sync_eventtype_regex_et,
        sync_heog_ch = config.sync_heog_ch,
        sync_et_ch = config.sync_et_ch,
        sync_heog_highpass = config.sync_heog_highpass,
        sync_heog_lowpass = config.sync_heog_lowpass,
        sync_plot_samps = config.sync_plot_samps,
        sync_calibration_string = config.sync_calibration_string,
        processing= "filt" if config.regress_artifact is None else "regress",
        _raw_split_size=config._raw_split_size,

        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Sync Eyelink."""
    if not config.sync_eyelink:
        msg = "Skipping, sync_eyelink is set to False …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return


    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(sync_eyelink, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)