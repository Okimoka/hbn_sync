% Terrible workaround that is needed for parsesmi to work with linux line endings
% Written by an LLM
function write_with_crlf(filename, lines)
    if isstring(lines)
        lines = cellstr(lines);
    elseif ischar(lines)
        lines = cellstr(string(lines));
    elseif iscellstr(lines)
        % ok
    elseif iscell(lines)
        lines = cellfun(@char, lines, 'UniformOutput', false);
    else
        error('write_with_crlf:InvalidInput', 'lines must be string, char, or cellstr.');
    end

    fid = fopen(filename, 'w', 'n', 'UTF-8');
    assert(fid > 0, 'write_with_crlf:IOError', 'Could not open %s for writing.', filename);
    c = onCleanup(@() fclose(fid));

    % Write each line with CRLF
    for k = 1:numel(lines)
        fprintf(fid, '%s\r\n', lines{k});
    end
end