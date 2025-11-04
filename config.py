import mne

bids_root = "extractedDataset"
deriv_root = "extractedDataset/derivs"
subjects_dir = None
#subjects = ["NDARAB678VYW","NDARDC504KWE","NDARDL033XRG","NDARTK720LER","NDARDZ794ZVP"] #"all" #["NDARAG429CGW"]
#subjects = ["NDARDZ794ZVP"] #"all" #["NDARAG429CGW"]
subjects = ["NDARAB678VYW"] ##["NDARKM301DY0"] #"all"
#subjects = ["NDARAF535XK6"]

ch_types = ["eeg"]
interactive = False
sessions = "all"
task = "symbolSearch"
task_et = "WISC_ProcSpeed"

task_is_rest = True
runs = ["1"]
et_has_run = False
et_has_task = True

epochs_tmin = 0
#rest_epochs_duration = 10
#rest_epochs_overlap = 0
baseline = None
#baseline: tuple[float | None, float | None] | None = (-0.2, 0)

#raw_resample_sfreq: float | None = 250

eeg_reference = "average"

ica_l_freq = 1 # ?

# determined by icalabel
l_freq: float | None = 1
h_freq: float | None = 100
ica_h_freq: float | None = 100

# data was recorded in the US
notch_freq = 60

on_error = "continue"


######### Remove these when doing Unfold analysis! ############

# positive / negative feedback
#conditions = ["HAPPY", "SAD"]
########conditions = ["# Message: 12XX", "# Message: 13XX"]

########epochs_tmin: float = -0.5
########epochs_tmax: float = 2.6 # since feedback is so infrequent, long ########epochs are okay
########
########baseline: tuple[float | None, float | None] | None = (-0.2, 0)
###############################################################



spatial_filter = "ica"
# ica_n_components = 96 ?
ica_algorithm = "picard-extended_infomax"
#ica_use_ecg_detection: bool = True
#ica_use_eog_detection: bool = True
ica_use_icalabel = True
#ica_reject: dict[str, float] | Literal["autoreject_local"] | None = "autoreject_local"

ica_reject = "autoreject_local" #TESTING
reject = "autoreject_local" #TESTING

#These are identical, just ensuring compatibility
sync_eyelink = True
sync_eye = True

#sync_eventtype_regex = "\\d-trigger=10 Image moves"
#sync_eventtype_regex_et = "trigger=10 Image moves"

#Contrast detection
#sync_eventtype_regex     = r"contrastTrial_start"
#sync_eventtype_regex_et  = r"# Message: 15"

sync_eventtype_regex     = r"(?:trialResponse|newPage)" #r"trialResponse"
sync_eventtype_regex_et  = r"# Message: (?:14|20)" #r"# Message: 14"

#sync_eventtype_regex     = r"trialResponse"
#sync_eventtype_regex_et  = r"# Message: 14"


#eog_channels = ["HEOGL", "HEOGR", "VEOGL", "VEOGU"]

#eeg_bipolar_channels = {"HEOG": ("HEOGL", "HEOGR"), "VEOG": ("VEOGL", "VEOGU")}
#eog_channels = ["HEOG", "VEOG"]
#sync_heog_ch = ("HEOG")


#eeg_bipolar_channels = {"HEOG": ("E40", "E109"),
#                            "VEOG": ("E21",  "E127")} #left eye
##eeg_bipolar_channels = {
##    #"HEOG": ("E127", "E126"),
##    "HEOG": ("E126", "E127"),
##    "VEOG": ("E22", "E127"), #left eye
##}


eeg_bipolar_channels = {
    #"HEOG": ("E127", "E126"),

    # Version 1, doesn't work well
    #### "HEOG": ("E127", "E126"), 
    #### "VEOG": ("E22", "E127"), #left eye

    # Version 2, works well but not sure why
    ####"HEOG": ("E109", "E40"),
    ####"VEOG": ("E22", "E127"), 

    # Version 3, seems to work well?
    "HEOG": ("E2", "E26"),
    "VEOG": ("E3", "E8")
}


eog_channels = ["HEOG", "VEOG"]

sync_heog_ch = "HEOG"

sync_et_ch = ("L POR X [px]", "R POR X [px]")

#sync_et_ch = "xpos_right"
sync_plot_samps = 3000

decode: bool = False
run_source_estimation = False

montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

eeg_template_montage = montage
drop_channels = ["Cz"]

n_jobs = 6