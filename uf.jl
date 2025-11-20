#TODO! Put this into the Julia template repo

using UnfoldBIDS
using Unfold
using LazyArtifacts

using Parquet, DataFrames, Tables

using Printf
using PythonCall
using CairoMakie # For plotting
using Statistics

#bids_root  = "/pfs/work9/workspace/scratch/st_st156392-mydata/mergedDataset"
bids_root  = "/home/oki/Desktop/hbn_sync-main/extractedDataset"

deriv_root = joinpath(bids_root, "derivatives")
analyzed_subject = "NDARUC804LKP"


# Find subjects that have a derivatives/ folder
processed_subs = filter(name -> startswith(name, "sub-") &&
                                 isdir(joinpath(deriv_root, name)),
                        readdir(deriv_root))

@info "Found $(length(processed_subs)) processed subjects in derivatives" processed_subs


# TODO comment this better
# The renaming + symlinks have to be done in order to have UnfoldBIDS properly recognize the dataset
# renames intermediary .fif files to end with _eeg.fif
# and creates symlinks for the corresponding _events.tsv files from raw data
# (LLM code)
for sub in processed_subs
    raw_eeg_dir   = joinpath(bids_root,  sub, "eeg")
    deriv_eeg_dir = joinpath(deriv_root, sub, "eeg")

    if !isdir(deriv_eeg_dir)
        @warn "No derivatives eeg dir for $sub, skipping" deriv_eeg_dir
        continue
    end
    if !isdir(raw_eeg_dir)
        @warn "No raw eeg dir for $sub, skipping symlink creation" raw_eeg_dir
        continue
    end

    # 1) Remove current event symlinks in derivatives/eeg
    for fname in readdir(deriv_eeg_dir)
        full = joinpath(deriv_eeg_dir, fname)
        if islink(full) && endswith(fname, "_events.tsv")
            rm(full; force = true)
            @info "Removed events symlink" full
        end
    end

    # 2) Raw events file for this subject & run-1 (symbolSearch)
    raw_events_name = "$(sub)_task-symbolSearch_run-1_events.tsv"
    raw_events_path = joinpath(raw_eeg_dir, raw_events_name)

    if !isfile(raw_events_path)
        @warn "Raw events file not found for $sub, skipping" raw_events_path
        continue
    end

    # 3) Process all .fif files with "run-1" in the name
    fif_files = filter(fname ->
        occursin("run-1", fname) && endswith(lowercase(fname), ".fif"),
        readdir(deriv_eeg_dir)
    )

    if isempty(fif_files)
        @info "No .fif files with run-1 for $sub in derivatives" deriv_eeg_dir
        continue
    end

    for fname in fif_files
        old_path = joinpath(deriv_eeg_dir, fname)
        new_fname = fname

        # If it doesn't end with _eeg.fif, append _eeg before the extension
        if !endswith(fname, "_eeg.fif")
            base, ext = splitext(fname)       # ext == ".fif"
            new_fname = base * "_eeg" * ext   # -> ..._eeg.fif
            new_path  = joinpath(deriv_eeg_dir, new_fname)

            if old_path != new_path
                if ispath(new_path)
                    @warn "Target FIF already exists, not renaming" old_path new_path
                else
                    mv(old_path, new_path)
                    @info "Renamed FIF" old_path new_path
                end
            end
        else
            new_path = old_path
        end

        # 4) Create events symlink: same name but _events.tsv instead of _eeg.fif
        events_fname = replace(new_fname, "_eeg.fif" => "_events.tsv")
        events_path  = joinpath(deriv_eeg_dir, events_fname)

        if ispath(events_path)
            @info "Events file already exists, skipping" events_path
            continue
        end

        symlink(raw_events_path, events_path)
        @info "Created events symlink" raw_events_path events_path
    end
end



layout_df = bids_layout(bids_root, derivatives=true, task="symbolSearch", run="1")

#only use proc-clean_raw_eeg.fif (not epoched, cleaned with ica)
df2 = filter(row ->
    endswith(row.file, "proc-clean_raw_eeg.fif") &&
    row.subject == analyzed_subject,
    layout_df
)

data_df = load_bids_eeg_data(df2)

raw_mne = data_df.raw[1]
eeg_data = PyArray(raw_mne.get_data()) .* 1e6
sfreq = pyconvert(Float64, raw_mne.info["sfreq"])


ann_pd = raw_mne.annotations.to_data_frame()
np = pyimport("numpy")

# convert datetime64[ns] → int64 nanoseconds → seconds since Unix epoch
ann_pd["onset"] = ann_pd["onset"].astype("int64") / np.int64(1_000_000_000)

ann_df  = DataFrame(PyTable(ann_pd))

#only use left eye saccades and fixations
#filter!(row -> !occursin('R', row.description), ann_df)
filter!(:description => x -> x in ("Fixation L", "Saccade L"), ann_df)
sort!(ann_df, :onset)  # ascending is the default

descs = ann_df.description

#alternating saccade L and fixation L
no_repeats = all(descs[i] != descs[i+1] for i in 1:length(descs)-1)

@assert no_repeats "Descriptions must alternate between 'Saccade L' and 'Fixation L'"
@assert ann_df.description[1] == "Fixation L" "First row must be 'Fixation L'"


fix_idx = findall(==("Fixation L"), ann_df.description)

merged_df = DataFrame(
    latency        = round.(Int, ann_df.onset[fix_idx] .* sfreq),
    type           = fill("fixation", length(fix_idx)),
    onset_sec      = ann_df.onset[fix_idx],
    fix_duration   = ann_df.duration[fix_idx],
    sacc_amplitude = ann_df.extra_5[fix_idx .+ 1],
)

# copy any other columns from the fixation rows
other_cols = setdiff(names(ann_df), ["onset", "duration", "description", "extra_5"])
for col in other_cols
    merged_df[!, col] = ann_df[fix_idx, col]
end

##########

fix = deepcopy(merged_df)

#TODO shouldnt be necessary if done correctly in python
rename!(fix, Dict(
    "latency"       => :latency,
    "sacc_amplitude"=> :sacc_amplitude,
    "onset_sec"     => :onset_sec,
    "fix_duration"  => :fix_duration,
    "extra_1"       => :fixation_position_x,
    "extra_2"       => :fixation_position_y,
    "extra_3"       => :fixation_position_end_x,
    "extra_4"       => :fixation_position_end_y,
))

fix.type = fill("fixation", nrow(fix))

# remove out of bounds fixations
minlat = round(Int, 1 - tmin * sfreq)
maxlat = round(Int, n_samp - tmax * sfreq)
filter!(r -> minlat ≤ r.latency ≤ maxlat, fix)

select!(fix, [:latency, :sacc_amplitude, :fixation_position_x, :fixation_position_y])



tmin = -0.2
tmax = 0.6
basis_fix = firbasis(τ = (tmin, tmax), sfreq = sfreq)

#takes an insane amount of time
#f_fix = @formula(
#    0 ~ 1 +
#        spl(fixation_position_x, 5) +
#        spl(fixation_position_y, 5) +
#        spl(sacc_amplitude, 5)
#)

# even this takes ~25 minutes
f_fix = @formula(0 ~ 1)

design = [Any => (f_fix, basis_fix)]

uf = fit(UnfoldModel, design, fix, eeg_data)

#some plots can be found in img
#don't look too promising, only intercept plot seems to be going into the right direction