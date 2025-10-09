%function subids = listSubfolders(p)
%    d = dir(p);
%    subids = {d([d.isdir]).name};
%    subids = setdiff(subids, {'.','..'});
%    subids = sort(subids);
%end


function subids = listSubfolders(p)
    d = dir(p);

    % Only directories, excluding '.' and '..'
    isSub  = [d.isdir];
    names  = {d.name};
    notDot = ~ismember(names, {'.','..'});

    % Age in minutes: (now - datenum) gives days → convert to minutes
    ageMinutes = (now - [d.datenum]) * 24 * 60;

    % Keep subfolders that are >= 30 minutes old
    keep = isSub & notDot & ageMinutes >= 30;

    subids = sort(names(keep));
end