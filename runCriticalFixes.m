function runCriticalFixes(varargin)
%RUNCRITICALFIXES Launch unified critical-fixes validation (review checklist).
    quick = nargin >= 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'quick');
    repoRoot = fileparts(mfilename('fullpath'));
    studyDir = fullfile(repoRoot, 'staleness_study');
    cmd = sprintf('cd "%s" && python3 run_critical_fixes.py', studyDir);
    if quick, cmd = [cmd ' --quick']; end
    fprintf('Running: %s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('Critical fixes run failed (exit %d).', status);
    end
    fprintf('Outputs in %s/critical_fixes_output/\n', studyDir);
end
