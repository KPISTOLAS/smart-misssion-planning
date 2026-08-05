% RUNPLANNEREVALUATION  Launch the Python planner evaluation study (Section V).
%
% Compares IUEF-EM against DARP, priority-TSP, lawnmower, potential-field,
% and systematic ablations under N=50 Monte-Carlo trials.
%
% Usage:
%   runPlannerEvaluation           % full study
%   runPlannerEvaluation('quick')  % smoke test
%
% See staleness_study/PLANNER_EVALUATION.md for details.

function runPlannerEvaluation(mode)
    if nargin < 1
        mode = 'full';
    end

    projRoot = fileparts(mfilename('fullpath'));
    studyDir = fullfile(projRoot, 'staleness_study');

    switch lower(mode)
        case 'quick'
            cmd = sprintf('cd "%s" && python3 run_planner_evaluation.py --quick', studyDir);
        case 'full'
            cmd = sprintf('cd "%s" && python3 run_planner_evaluation.py --n-mc 50', studyDir);
        otherwise
            error('mode must be ''full'' or ''quick''');
    end

    fprintf('Running planner evaluation...\n  %s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('Planner evaluation failed. Install deps: pip install -r requirements.txt');
    end
    fprintf('Outputs: %s/planner_output/\n', studyDir);
end
