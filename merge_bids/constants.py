from pathlib import Path

# Expected participants.tsv columns, in order
PARTICIPANTS_COLUMNS = [
    "participant_id","release_number","sex","age","ehq_total","commercial_use","full_pheno",
    "p_factor","attention","internalizing","externalizing","RestingState","DespicableMe",
    "FunwithFractals","ThePresent","DiaryOfAWimpyKid","contrastChangeDetection_1",
    "contrastChangeDetection_2","contrastChangeDetection_3","surroundSupp_1","surroundSupp_2",
    "seqLearning6target","seqLearning8target","symbolSearch"
]




# Root-level JSON files that must exist in every release
# Some tasks don't have root level files even though maybe they should?
REQUIRED_ROOT_JSON = [
    "participants.json",
    "task-contrastChangeDetection_eeg.json",
    "task-contrastChangeDetection_events.json",
    "task-DespicableMe_eeg.json",
    "task-DespicableMe_events.json",
    "task-DiaryOfAWimpyKid_eeg.json",
    "task-DiaryOfAWimpyKid_events.json",
    "task-FunwithFractals_eeg.json",
    "task-FunwithFractals_events.json",
    "task-RestingState_eeg.json",
    "task-RestingState_events.json",
    #"task-seqLearning6target_eeg.json",
    #"task-seqLearning6target_events.json",
    #"task-seqLearning8target_eeg.json",
    #"task-seqLearning8target_events.json",
    "task-surroundSupp_eeg.json",
    "task-surroundSupp_events.json",
    "task-symbolSearch_eeg.json",
    "task-symbolSearch_events.json",
    "task-ThePresent_eeg.json",
    "task-ThePresent_events.json",
]

# The "code" directory must exist and contain exactly these TSVs
REQUIRED_CODE_TSV = [
    "contrastChangeDetection_1_quality_table.tsv",
    "contrastChangeDetection_2_quality_table.tsv",
    "contrastChangeDetection_3_quality_table.tsv",
    "DespicableMe_quality_table.tsv",
    "DiaryOfAWimpyKid_quality_table.tsv",
    "FunwithFractals_quality_table.tsv",
    "RestingState_quality_table.tsv",
    "seqLearning6target_quality_table.tsv",
    "seqLearning8target_quality_table.tsv",
    "surroundSupp_1_quality_table.tsv",
    "surroundSupp_2_quality_table.tsv",
    "symbolSearch_quality_table.tsv",
    "ThePresent_quality_table.tsv",
    "seqLearning_quality_table.tsv",
]

# Quality table required columns (and order)
QUALITY_TABLE_COLUMNS = ["Row","data_pnts","event_cnt","key_events_exist","quality_checks"]

# Allowed/expected root entries (files and directories). sub-* folders are also allowed.
ALLOWED_ROOT_BASENAMES = {
    "participants.tsv",
    "dataset_description.json",
    "CHANGES",
    "README",
    "code",
    *REQUIRED_ROOT_JSON
}

# filenames allowed inside each subject's eeg/ directory.
ALLOWED_EEG_PATTERNS = {
"sub-{ID}_task-DespicableMe_channels.tsv",
"sub-{ID}_task-DespicableMe_eeg.json",
"sub-{ID}_task-DespicableMe_eeg.set",
"sub-{ID}_task-DespicableMe_events.tsv",
"sub-{ID}_task-DiaryOfAWimpyKid_channels.tsv",
"sub-{ID}_task-DiaryOfAWimpyKid_eeg.json",
"sub-{ID}_task-DiaryOfAWimpyKid_eeg.set",
"sub-{ID}_task-DiaryOfAWimpyKid_events.tsv",
"sub-{ID}_task-FunwithFractals_channels.tsv",
"sub-{ID}_task-FunwithFractals_eeg.json",
"sub-{ID}_task-FunwithFractals_eeg.set",
"sub-{ID}_task-FunwithFractals_events.tsv",
"sub-{ID}_task-RestingState_channels.tsv",
"sub-{ID}_task-RestingState_eeg.json",
"sub-{ID}_task-RestingState_eeg.set",
"sub-{ID}_task-RestingState_events.tsv",
"sub-{ID}_task-ThePresent_channels.tsv",
"sub-{ID}_task-ThePresent_eeg.json",
"sub-{ID}_task-ThePresent_eeg.set",
"sub-{ID}_task-ThePresent_events.tsv",
"sub-{ID}_task-contrastChangeDetection_run-1_channels.tsv",
"sub-{ID}_task-contrastChangeDetection_run-1_eeg.json",
"sub-{ID}_task-contrastChangeDetection_run-1_eeg.set",
"sub-{ID}_task-contrastChangeDetection_run-1_events.tsv",
"sub-{ID}_task-contrastChangeDetection_run-2_channels.tsv",
"sub-{ID}_task-contrastChangeDetection_run-2_eeg.json",
"sub-{ID}_task-contrastChangeDetection_run-2_eeg.set",
"sub-{ID}_task-contrastChangeDetection_run-2_events.tsv",
"sub-{ID}_task-contrastChangeDetection_run-3_channels.tsv",
"sub-{ID}_task-contrastChangeDetection_run-3_eeg.json",
"sub-{ID}_task-contrastChangeDetection_run-3_eeg.set",
"sub-{ID}_task-contrastChangeDetection_run-3_events.tsv",
"sub-{ID}_task-seqLearning6target_channels.tsv",
"sub-{ID}_task-seqLearning6target_eeg.json",
"sub-{ID}_task-seqLearning6target_eeg.set",
"sub-{ID}_task-seqLearning6target_events.tsv",
"sub-{ID}_task-seqLearning8target_channels.tsv",
"sub-{ID}_task-seqLearning8target_eeg.json",
"sub-{ID}_task-seqLearning8target_eeg.set",
"sub-{ID}_task-seqLearning8target_events.tsv",
"sub-{ID}_task-surroundSupp_run-1_channels.tsv",
"sub-{ID}_task-surroundSupp_run-1_eeg.json",
"sub-{ID}_task-surroundSupp_run-1_eeg.set",
"sub-{ID}_task-surroundSupp_run-1_events.tsv",
"sub-{ID}_task-surroundSupp_run-2_channels.tsv",
"sub-{ID}_task-surroundSupp_run-2_eeg.json",
"sub-{ID}_task-surroundSupp_run-2_eeg.set",
"sub-{ID}_task-surroundSupp_run-2_events.tsv",
"sub-{ID}_task-symbolSearch_channels.tsv",
"sub-{ID}_task-symbolSearch_eeg.json",
"sub-{ID}_task-symbolSearch_eeg.set",
"sub-{ID}_task-symbolSearch_events.tsv"
}

ET_TO_EEG_TASK = {
    "resting": "RestingState",
    "NODOOR_resting": "RestingState",
    "Video1": "DespicableMe",
    "Video2": "DiaryOfAWimpyKid",
    "Video3": "FunwithFractals",
    "Video4": "ThePresent",
    "Video-DM": "DespicableMe",
    "Video-WK": "DiaryOfAWimpyKid",
    "Video-FF": "FunwithFractals",
    "Video-TP": "ThePresent",
    "WISC_ProcSpeed": "symbolSearch",
    "NODOOR_WISC_ProcSpeed": "symbolSearch",
    "WISC_2nd": None,
    "SAIIT_2AFC_Block1": "contrastChangeDetection_run-1",
    "SAIIT_2AFC_Block2": "contrastChangeDetection_run-2",
    "SAIIT_2AFC_Block3": "contrastChangeDetection_run-3",
    "NODOOR_SAIIT_2AFC_Block1": "contrastChangeDetection_run-1",
    "NODOOR_SAIIT_2AFC_Block2": "contrastChangeDetection_run-2",
    "SurrSupp_Block1": "surroundSupp_run-1",
    "SurrSupp_Block2": "surroundSupp_run-2",
    "NODOOR_SurrSupp_Block1": "surroundSupp_run-1",
    "NODOOR_SurrSupp_Block2": "surroundSupp_run-2",
    "II_SurrSupp_Block1": "surroundSupp_run-1",
    "vis_learn": None,             # has to be assigned "seqLearning6target" or "seqLearning8target"
    "NODOOR_vis_learn": None,      # same
    "calibration": None,
}





ET_SUBDIRS_ALLOWED = {"txt", "tsv", "idf"}


TEMPLATE_README = """# The HBN-EEG Dataset
This is a merged dataset of multiple HBN-EEG releases. See the original release datasets for more information: https://neuromechanist.github.io/data/hbn/"""


def is_subject_dir(name: str) -> bool:
    # Accept sub-<id> subject directories at the root of each release
    return name.startswith("sub-") and len(name) > 4
