function overview_by_task = run_batch_et_eeg(taskids, taskids2, bids_paths, et_path, sync_codes, degrees_per_pixel, screen_res, overview_filename)

% Inputs
%   taskids         - cellstr of task names used in EEG filenames (e.g., {'symbolSearch','...'})
%   taskids2        - cellstr aliases used in ET filenames, same length/order as taskids (e.g., {'ProcSpeed','...'})
%   bids_paths      - cell array of paths to BIDS roots containing sub-<subid>/eeg/*.set
%   et_path         - path to ET root containing <subid>/Eyetracking/txt/*.txt
%   overview_filename (optional) - output workbook name; default 'overview.xlsx'
%
% Output
%   overview_by_task - struct of tables per task (also written to Excel, one sheet per task)
%
% Notes
% - Requires EEGLAB + EYE-EEG on the MATLAB path.

%TODO
% Support different channel amounts than 43 (for pop_importeyetracker, but also to use correct indexes)
% Support subjects that have ET data as tsv / idt rather than txt

%Remove subjects that have no task where both eeg and et are simultaneously present for that task

    if nargin < 6 || isempty(overview_filename)
        overview_filename = 'overview.xlsx';
    end

    % Suppress all figure popups of results
    prevVis = get(groot,'DefaultFigureVisible');
    set(groot,'DefaultFigureVisible','off');
    cleanupFigVis = onCleanup(@() set(groot,'DefaultFigureVisible', prevVis));

    % Init EEGLAB
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui'); %#ok<ASGLU>

    % et_path contains all subjects, bids_path only contains a subset
    % creates a sorted list of all subject ids
    subids = helpers.listSubfolders(et_path);
    eeg_index = build_eeg_index(bids_paths);

    % init resulting struct
    overview_by_task = struct;

    % Per task
    for t = 1:numel(taskids)
        taskid  = taskids{t}; %e.g. FunwithFractals
        taskid2 = taskids2{t}; %e.g. Video-FF
        sheet   = taskid;

        % Pre-load table for looking up subject rows later
        try
            T = readtable(overview_filename, 'Sheet', sheet, 'TextType','char');
            % If earlier runs created "Var1", rename it to "subject_id"
            if ~ismember('subject_id', T.Properties.VariableNames) && ~isempty(T)
                T.Properties.VariableNames{1} = 'subject_id';
            end
            % Ensure subject_id is cellstr
            if ~isempty(T) && isstring(T.subject_id)
                T.subject_id = cellstr(T.subject_id);
            end
        catch
            T = table();  % no sheet yet
        end
        
        % existing ids as cellstr

        existing_ids_cell = {};
        if ~isempty(T)
            existing_ids_cell = T.subject_id(:);
        end

        % subids already exists (created before the task loop)
        % Only process subjects not already present; keep original order
        to_process = setdiff(subids, existing_ids_cell, 'stable');

        disp(['Processing task ' taskid ' (' num2str(t) '/' num2str(numel(taskids)) '), ' num2str(numel(to_process)) ' subjects to process.']);

        n = numel(to_process);
        has_eeg   = false(n,1);
        has_et    = false(n,1);

        % Per subject
        for i = 1:n
            subid = to_process{i};

            EEG = [];
            ET  = [];

            % Skip if this subject is already logged (exists at all)
            %if any(existing_ids == string(subid))
            %    fprintf('Skipping %s for task %s: already in sheet.\n', subid, taskid);
            %    continue;
            %end

            key      = [subid '|' taskid];
            if isKey(eeg_index, key)
                eeg_file = eeg_index(key);
            else
                eeg_file = '';
            end

            et_file  = fullfile(et_path, subid, 'txt', sprintf('%s_%s_Samples.txt', subid, taskid2));

            has_eeg(i) = ~isempty(eeg_file);
            has_et(i)  = exist(et_file,'file') == 2;

            %disp([subid ': has EEG: ' num2str(has_eeg(i)) ', has ET: ' num2str(has_et(i))]);
            %disp(['  EEG: ' eeg_file]);
            %disp(['  ET:  ' et_file]);
            %disp(exist(et_file,'file'));

            % Can't sync data otherwise
            if has_eeg(i) && has_et(i)
                try
                    % Load EEG
                    EEG = pop_loadset('filename', eeg_file);
                    
                    if EEG.nbchan ~= 129
                        fprintf('Skipping subject %s for task %s: EEG.nbchan = %d\n', subid, taskid, EEG.nbchan);
                        EEG = []; ET = [];
                        continue;
                    end

                    % Overwrite the event names with event codes in the first column
                    % Else EYE-EEG does not recognize the events
                    [EEG.event.type] = EEG.event.event_code;

                    % Preprocess SMI file. The MSGs require some keyword
                    % Turns lines like these:
                    % 2537825103	MSG	1	# Message: 14
                    % Into
                    % 2537825103	MSG	1	# Message: SYNC 14
                    et_txt  = et_file;
                    tmp_txt = fullfile(tempdir, sprintf('%s_%s_Samples_sync.txt', subid, taskid2));
                    lines   = readlines(et_txt);
                    newLines= regexprep(lines, '(# Message:\s)(\d+)', '$1SYNC $2');
                    helpers.write_with_crlf(tmp_txt, newLines);
                    tmp_mat = fullfile(tempdir, sprintf('%s_%s_temp_et.mat', subid, taskid2));

                    % Function "Parse Eyetracker raw data > text file from SMI"
                    ET = parsesmi(char(tmp_txt), char(tmp_mat), 'SYNC'); %#ok<NASGU>

                    % Function "Import & Synchronize ET"
                    % Importing all channels by default
                    EEG = pop_importeyetracker( ...
                        EEG, tmp_mat, sync_codes{t}, [1:43], ...
                        {'Time','Trial','L-Raw-X-(px)','L-Raw-Y-(px)','R-Raw-X-(px)','R-Raw-Y-(px)', ...
                         'L-Dia-X-(px)','L-Dia-Y-(px)','L-Mapped-Diameter-(mm)','R-Dia-X-(px)', ...
                         'R-Dia-Y-(px)','R-Mapped-Diameter-(mm)','L-CR1-X-(px)','L-CR1-Y-(px)', ...
                         'L-CR2-X-(px)','L-CR2-Y-(px)','R-CR1-X-(px)','R-CR1-Y-(px)','R-CR2-X-(px)', ...
                         'R-CR2-Y-(px)','L-POR-X-(px)','L-POR-Y-(px)','R-POR-X-(px)','R-POR-Y-(px)', ...
                         'Timing','L-Validity','R-Validity','Pupil-Confidence','L-Plane','R-Plane', ...
                         'L-EPOS-X','L-EPOS-Y','L-EPOS-Z','R-EPOS-X','R-EPOS-Y','R-EPOS-Z', ...
                         'L-GVEC-X','L-GVEC-Y','L-GVEC-Z','R-GVEC-X','R-GVEC-Y','R-GVEC-Z','Trigger'}, ...
                        0,1,0,1,4);

                    % Function "Reject data based on eye track > Reject bad cont. data"
                    % Channels 150 - 153 correspond to L-POR-X, L-POR-Y, R-POR-X, R-POR-Y
                    % TODO Calibration of ET files says screen size is 1440 x 900, but paradigm paper says 800 x 600 ?
                    % EEG.event is broken by this step. Maybe use a backup?
                    EEG = pop_rej_eyecontin(EEG, 150:153, [1 1 1 1], [screen_res(1) screen_res(2) screen_res(1) screen_res(2)], 50, 1);

                    % Function "Special-purpose functions > Evaluate ET/EEG synchronization (cross-corr.)"
                    % channel 150 is L-POR-X (horiz. gaze position)
                    % channel 128 (E128) and 125 (E125) are the left and right EOG electrodes for GSN-HydroCel-128
                    EEG = pop_checksync(EEG, 150, 128, 125, 1);

                    % Function "Detect saccades & fixations"
                    % TODO compare how well these align with the fixations and saccades listed in
                    % <subjectid>_WISC_ProcSpeed_Events.txt
                    % Claims to add events into EEG.event, but they don't seem to be there?
                    % Saccade duration chosen as 2 samples @ 60Hz (4 ms)
                    % Clustering mode 2 chosen because EYE-EEG detected clusters
                    EEG = pop_detecteyemovements(EEG, [150 151], [152 153], 6, 2, degrees_per_pixel, 1, 0, 25, 2, 1, 0, 0);


                catch ME
                    warning('Failed for sub %s, task %s: %s', subid, taskid, ME.message);
                    EEG = []; ET = [];
                    % leave metrics as NaN for this subject
                end

                % --- Collect metrics (if available) and write/update the row immediately ---
                metricsT = extract_metrics_simple(EEG, ET);  % returns 1xN table or empty

                % --- Upsert row now ---
                rowT = table({subid}, double(has_eeg(i)), double(has_et(i)), ...
                    'VariableNames', {'subject_id','has_eeg','has_et'});
                if ~isempty(metricsT), rowT = [rowT metricsT]; end

                upsert_row_to_sheet(overview_filename, sheet, rowT);
                %existing_ids(end+1,1) = string(subid);

            end


        end

        % Assemble and return full table for this task
        try
            T = readtable(overview_filename, 'Sheet', sheet);
        catch
            T = table();
        end
        overview_by_task.(matlab.lang.makeValidName(taskid)) = T;

    end

    fprintf('Wrote/updated overview per subject to %s\n', overview_filename);
end





% Helper functions largely written by LLMs


% Helper function for allowing updates of the overview every time a subject is processed
function upsert_row_to_sheet(filename, sheetname, rowT)
    % Read existing or start empty
    try
        T = readtable(filename, 'Sheet', sheetname);
    catch
        T = rowT([],:); % empty with same vars as rowT
    end

    % Normalize id types
    if isstring(rowT.subject_id), rowT.subject_id = cellstr(rowT.subject_id); end
    if ~isempty(T) && isstring(T.subject_id), T.subject_id = cellstr(T.subject_id); end

    % Add any new columns from rowT to T (metrics are numeric; fill with NaN)
    newVars = setdiff(rowT.Properties.VariableNames, T.Properties.VariableNames);
    for i = 1:numel(newVars)
        vn = newVars{i};
        T.(vn) = NaN(height(T),1);
    end

    % Ensure rowT has any columns that already exist in T
    missingInRow = setdiff(T.Properties.VariableNames, rowT.Properties.VariableNames);
    for i = 1:numel(missingInRow)
        vn = missingInRow{i};
        if isnumeric(T.(vn))
            rowT.(vn) = NaN;
        else
            % e.g., subject_id is text; add empty placeholder
            rowT.(vn) = {''};
        end
    end

    % Reorder rowT to T’s column order
    rowT = rowT(:, T.Properties.VariableNames);

    % Upsert by subject_id
    idx = find(strcmp(T.subject_id, rowT.subject_id{1}), 1);
    if isempty(idx)
        T = [T; rowT];
    else
        T(idx,:) = rowT;
    end

    writetable(T, filename, 'Sheet', sheetname);
end




function metricsT = extract_metrics_simple(EEG, ET)
    m = struct();

    if ~isempty(ET) && isfield(ET,'etc') && isfield(ET.etc,'parsesmi_quality')
        S = ET.etc.parsesmi_quality;
        fn = fieldnames(S);
        for k = 1:numel(fn), m.(fn{k}) = double(S.(fn{k})); end
    end

    if ~isempty(EEG) && isfield(EEG,'etc')
        src = {'eyetracker_syncmetrics','rej_eyecontin_quality','checksync_quality','eyemovements_quality'};
        for j = 1:numel(src)
            if isfield(EEG.etc, src{j})
                S  = EEG.etc.(src{j});
                fn = fieldnames(S);
                for k = 1:numel(fn), m.(fn{k}) = double(S.(fn{k})); end
            end
        end
    end

    if isempty(fieldnames(m))
        metricsT = table();
    else
        metricsT = struct2table(m, 'AsArray', true);
    end
end


function eeg_index = build_eeg_index(bids_paths)
%BUILD_EEG_INDEX  Scan multiple BIDS roots once and index EEG .set files.
%   eeg_index = build_eeg_index(bids_paths)
%   - bids_paths: char, string, string array, or cellstr of BIDS roots.
%   - Returns containers.Map with keys "subid|task" and values = full paths.
%
%   If the same subject+task exists in multiple roots, the FIRST root wins
%   (based on the order you pass in).

    % Normalize input to cellstr row
    if ischar(bids_paths) || (isstring(bids_paths) && isscalar(bids_paths))
        bids_paths = cellstr(bids_paths);
    elseif isstring(bids_paths)
        bids_paths = cellstr(bids_paths(:).'); % row
    elseif ~iscell(bids_paths)
        error('bids_path must be char, string, string array, or cell array of char/strings.');
    end

    eeg_index = containers.Map('KeyType','char','ValueType','char');

    t0 = tic;
    total_added = 0;

    % We only scan the canonical BIDS path pattern to avoid huge recursive walks
    % Pattern: <root>/sub-*/eeg/sub-*_task-*_eeg.set
    for r = 1:numel(bids_paths)
        root = bids_paths{r};
        d = dir(fullfile(root, 'sub-*', 'eeg', 'sub-*_task-*_eeg.set'));
        for k = 1:numel(d)
            name = d(k).name;
            toks = regexp(name, '^sub-([^-_]+)_task-([^-_]+)_eeg\.set$', 'tokens', 'once');
            if isempty(toks), continue; end
            subid  = toks{1};
            taskid = toks{2};
            key    = [subid '|' taskid];
            if ~isKey(eeg_index, key)
                eeg_index(key) = fullfile(d(k).folder, d(k).name);
                total_added = total_added + 1;
            end
        end
    end

    fprintf('Indexed %d EEG files across %d BIDS roots in %.2fs.\n', ...
            total_added, numel(bids_paths), toc(t0));
end

