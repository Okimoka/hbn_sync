%{

For each release we need two versions of the dataset:
1. The original (EEG non-BIDS) dataset from nitrc.org (e.g. https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/downloads/downloads_EEG_R11.html)
2. The curated (EEG BIDS) dataset from nemar (e.g. https://nemar.org/dataexplorer/detail?dataset_id=ds005516)

Only the original dataset contains the Eye-Tracking data, and only the BIDS dataset has suitably formatted EEG data.
For this reason, the two versions of the dataset will be referred as "et" and "bids".


TODO:
Filter participants based on the participants.tsv from the bids dataset (remove subjects with "Caution" and "Unavailable")

%}

addpath '/home/oki/matlab/eeglab2024.2';

% Task names in BIDS EEG files
tasks   = {'symbolSearch', 'DiaryOfAWimpyKid', 'ThePresent', 'FunwithFractals', 'DespicableMe'};
% Task names in original (ET) files
aliases = {'WISC_ProcSpeed', 'Video-WK', 'Video-TP', 'Video-FF', 'Video-DM'};

bidsroots    = {...
    '/media/oki/Beige/Release11/EEG', ...
    '/media/oki/Beige/Release10/EEG', ...
    '/media/oki/Beige/Release9/EEG', ...
    '/media/oki/Beige/Release8/EEG', ...
    '/media/oki/Beige/Release7/EEG', ...
    '/media/oki/Beige/Release6/EEG', ...
    '/media/oki/Beige/Release5/EEG', ...
    '/media/oki/Beige/Release4/EEG', ...
    '/media/oki/Beige/Release3/EEG', ...
    '/media/oki/Beige/Release2/EEG', ...
    '/media/oki/Beige/Release1/EEG'};

etroot  = '/media/oki/Beige/Eye-tracking-pull';

%{
Event codes of the events that will be used for synchronization
There needs to be at least one event at the start and one at the end

92 is the event_code for symbolSearch_start
20 is newPage which seems to always be the last event before boundary
8X and 10X are the video_start and video_stop events for all movies (X=1,2,3,4)
%}

sync_codes = {
    [92 20],   % symbolSearch
    [81 101],  % DiaryOfAWimpyKid
    [84 104],  % ThePresent
    [82 102],  % FunwithFractals
    [83 103]   % DespicableMe
};

screen_res = [1440 900]; % Screen resolution in pixels (width x height)
screen_size = [330 240]; % Screen size in mm (width x height)
viewing_dist = 700; % Viewing distance in mm

degrees_per_pixel = helpers.degPerPix(screen_size(1), screen_res(1), viewing_dist);

disp(degrees_per_pixel);

run_batch_et_eeg(tasks, aliases, bidsroots, etroot, sync_codes, degrees_per_pixel, screen_res, 'overview.xlsx');

