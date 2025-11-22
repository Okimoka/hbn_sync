import mne
import numpy as np
import matplotlib.pyplot as plt

subject = "NDARUC804LKP"
eye = "R"

fif_path = f"extractedDataset/derivatives/sub-{subject}/eeg/sub-{subject}_task-symbolSearch_run-1_proc-clean_raw_eeg.fif"

# Load raw
raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=True)

annot = raw.annotations
sfreq = raw.info["sfreq"]

# Only fixations of specified eye
desc_fix = f"Fixation {eye}"
mask_fix = annot.description == desc_fix
fix_onsets = annot.onset[mask_fix] # in seconds
onset_samples = (fix_onsets * sfreq).astype(int) # in samples
fix_durations = annot.duration[mask_fix] # in seconds

print(f"Number of fixations ({desc_fix}):", len(fix_onsets))

# Use max fixation length as plot window
win_sec = float(np.max(fix_durations))
print(f"Using window length of {win_sec*1000:.1f} ms")



# List of [sample, 0, event_code] for epoch creation
event_code = 1
events = np.column_stack([
    onset_samples,
    np.zeros_like(onset_samples, dtype=int),
    np.full_like(onset_samples, event_code, dtype=int),
])
event_id = {desc_fix: event_code}


tmin = -0.1
tmax = win_sec

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(tmin, 0.0),
    preload=True,
    reject_by_annotation=True,
)



# Average across all fixation epochs
evoked = epochs[desc_fix].average()


# Volt threshold to filter bad channels
thresh_V = 20.0 * 1e-6

data = evoked.get_data()  # (n_channels, n_times)
max_per_channel = np.max(np.abs(data), axis=1)

bad_idx = np.where(max_per_channel > thresh_V)[0]
bad_chs = [evoked.ch_names[i] for i in bad_idx]

print("Channels exceeding threshold:", bad_chs)

# mark as bad
evoked.info["bads"] = bad_chs
evoked.plot(exclude="bads")


#### Plot mean of all good channels ####

picks_good = mne.pick_types(
    evoked.info,
    eeg=True,
    meg=False,
    eog=False,
    stim=False,
    exclude="bads",
)

mean_frp = evoked.data[picks_good].mean(axis=0)  # in Volts

plt.figure()
plt.plot(evoked.times, mean_frp * 1e6)  # convert to µV
plt.axvline(0, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title(f"Mean FRP across channels ({desc_fix}, |amp| < {thresh_V / 1e-6} µV)")
plt.tight_layout()
plt.show()