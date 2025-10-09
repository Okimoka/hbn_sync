from types import SimpleNamespace
from datetime import datetime, timezone
from collections import Counter
import mne
import os.path
import re
import json
import csv
import numpy as np
from mne_bids import BIDSPath

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

    ann_on, ann_du, ann_ds = [], [], [] #onset, duration, description
    t0 = times[0]

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
                #Looks like (no commas) "Fixation L, 1, 1, 2524661981, 2524811929, 149948, 1077.17, 697.01, 29, 17, -1, 12.61, 12.61
                elif len(parts) >= 5 and (kind.startswith("Fixation") or kind.startswith("Saccade") or kind.startswith("Blink")):
                    start, end = float(parts[3]), float(parts[4])
                    ann_on.append((start - t0) / 1e6)
                    ann_du.append((end - start) / 1e6)
                    ann_ds.append(kind)
    else:
        logger.info(**gen_log_kwargs(message=f"SMI _Events.txt not found, using Events from _Samples.txt"))
        # Fallback: inline MSG rows as zero-duration annotations
        for t_us, desc in inline_msgs:
            ann_on.append((t_us - t0) / 1e6)
            ann_du.append(0.0)
            ann_ds.append(desc)

    if ann_on:
        raw_et.set_annotations(mne.Annotations(onset=ann_on, duration=ann_du, description=ann_ds))

    return raw_et



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
    from scipy.signal import correlate

    raw_fnames = [in_files.pop(f"raw_run-{run}") for run in cfg.runs]
    et_fnames = [in_files.pop(f"et_run-{run}") for run in cfg.runs]
    
    logger.info(**gen_log_kwargs(message=f"Found the following eye-tracking files: {et_fnames}"))
    out_files = dict()
    bids_basename = raw_fnames[0].copy().update(processing=None, split=None, run=None)
    out_files["eyelink"] = bids_basename.copy().update(processing="eyelink", suffix="raw")
    del bids_basename

    for idx, (run, raw_fname,et_fname) in enumerate(zip(cfg.runs, raw_fnames,et_fnames)):
        msg = f"Syncing Eyelink ({et_fname.basename}) and EEG data ({raw_fname.basename})."
        logger.info(**gen_log_kwargs(message=msg))
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

        et_format = et_fname.extension

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

            raw_et = _read_raw_iview(str(et_fname), event_fname)
        else:
            raise AssertionError("ET file is neither an `.asc`, `.edf`, nor `.txt`.")



        # If the user did not specify a regular expression for the eye-tracking sync events, it is assumed that it's
        # identical to the regex for the EEG sync events
        if not cfg.sync_eventtype_regex_et:
            cfg.sync_eventtype_regex_et = cfg.sync_eventtype_regex
        
        et_sync_times = [annotation["onset"] for annotation in raw_et.annotations if re.search(cfg.sync_eventtype_regex_et,annotation["description"])]
        sync_times    = [annotation["onset"] for annotation in raw.annotations    if re.search(cfg.sync_eventtype_regex,   annotation["description"])]
        assert len(et_sync_times) == len(sync_times),f"Detected eyetracking and EEG sync events were not of equal size ({len(et_sync_times)} vs {len(sync_times)}). Adjust your regular expressions via 'sync_eventtype_regex_et' and 'sync_eventtype_regex' accordingly"
        #logger.info(**gen_log_kwargs(message=f"{et_sync_times}"))
        #logger.info(**gen_log_kwargs(message=f"{sync_times}"))


        # Check whether the eye-tracking data contains nan values. If yes replace them with zeros.
        if np.isnan(raw_et.get_data()).any():

            # Set all nan values in the eye-tracking data to 0 (to make resampling possible)
            # TODO: Decide whether this is a good approach or whether interpolation (e.g. of blinks) is useful
            # TODO: Decide about setting the values (e.g. for blinks) back to nan after synchronising the signals
            np.nan_to_num(raw_et._data, copy=False, nan=0.0)
            logger.info(**gen_log_kwargs(message=f"The eye-tracking data contained nan values. They were replaced with zeros."))

        #mne.preprocessing.eyetracking.interpolate_blinks(raw_et, buffer=(0.05, 0.05), interpolate_gaze=True)       

        # Align the data
        mne.preprocessing.realign_raw(raw, raw_et, sync_times, et_sync_times)

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
        # cross correlate them
        corr = correlate(heog_array[0], et_array[0], mode="same") / heog_array.shape[1]
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

    axes[1,1].set_title("Raw - ET event onset distances histogram")
    axes[1,1].set_xlabel("Samples")
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