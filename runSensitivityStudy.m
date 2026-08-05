function runSensitivityStudy(varargin)
%RUNSENSITIVITYSTUDY Launch the Python EPCA-M sensitivity analysis study.
    quick = nargin >= 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'quick');
    repoRoot = fileparts(mfilename('fullpath'));
    studyDir = fullfile(repoRoot, 'staleness_study');
    cmd = sprintf('cd "%s" && python3 run_sensitivity_study.py', studyDir);
    if quick, cmd = [cmd ' --quick']; end
    fprintf('Running: %s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('Sensitivity study failed (exit %d).', status);
    end
    fprintf('Outputs in %s/sensitivity_output/\n', studyDir);
end
