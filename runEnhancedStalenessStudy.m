% RUNENHANCEDSTALENESSSTUDY  Launch the Python enhanced staleness Monte-Carlo study.
%
% The enhanced staleness / AoI model (stochastic NTN channel, calibrated map
% fade & ghost drift, adaptive sync, IUEF-EM planner integration) is implemented
% in Python for reproducibility and plotting.  This script invokes it from the
% MATLAB workspace.
%
% Prerequisites:
%   - Python 3.9+ with numpy, matplotlib (see staleness_study/requirements.txt)
%
% Usage:
%   runEnhancedStalenessStudy              % full study (N=50)
%   runEnhancedStalenessStudy('quick')     % fast smoke test
%
% Outputs are written to staleness_study/output/ (figures + JSON summary).

function runEnhancedStalenessStudy(mode)
    if nargin < 1
        mode = 'full';
    end

    projRoot = fileparts(mfilename('fullpath'));
    studyDir = fullfile(projRoot, 'staleness_study');

    if ~isfolder(studyDir)
        error('staleness_study folder not found at %s', studyDir);
    end

    switch lower(mode)
        case 'quick'
            cmd = sprintf('cd "%s" && python3 run_staleness_study.py --quick', studyDir);
        case 'full'
            cmd = sprintf('cd "%s" && python3 run_staleness_study.py --n-mc 50', studyDir);
        otherwise
            error('mode must be ''full'' or ''quick''');
    end

    fprintf('Running enhanced staleness study...\n  %s\n', cmd);
    status = system(cmd);

    if status ~= 0
        error('Python study failed (exit code %d). Check Python deps: pip install -r requirements.txt', status);
    end

    summaryPath = fullfile(studyDir, 'output', 'staleness_study_summary.json');
    if isfile(summaryPath)
        fprintf('Summary written to:\n  %s\n', summaryPath);
    end

    figDir = fullfile(studyDir, 'output');
    fprintf('Figures in:\n  %s\n', figDir);
end
