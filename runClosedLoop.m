function runClosedLoop(varargin)
%RUNCLOSEDLOOP Launch the Python closed-loop EPCA-M Monte-Carlo study.
%
%   Requires Python 3 with staleness_study/requirements.txt installed.
%
%   Usage (from repo root):
%       runClosedLoop              % full study N=50
%       runClosedLoop('quick')     % smoke test N=12
%
    quick = false;
    if nargin >= 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'quick')
        quick = true;
    end
    repoRoot = fileparts(mfilename('fullpath'));
    studyDir = fullfile(repoRoot, 'staleness_study');
    cmd = sprintf('cd "%s" && python3 run_closed_loop.py', studyDir);
    if quick
        cmd = [cmd ' --quick'];
    end
    fprintf('Running: %s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('Closed-loop study failed (exit %d).', status);
    end
    fprintf('Outputs in %s/closed_loop_output/\n', studyDir);
end
