% GEN_DRONE_TO_FIELD  Build the real PHM map + priority UAV plan and export the
% arrays needed to render a presentation figure of the drone moving to the field.

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

N  = 54;
M  = 72;
dx = 18;          % meters per cell edge
numUAV = 3;       % coordinated fleet ingressing the field

opts = struct('terrain', 'hilly', 'treeDensity', 'low', ...
    'alpha', 1.0, 'beta', 0.55, 'gamma', 0.32, 'rngSeed', 8148);

mapData = MapGenerator.build(N, M, dx, opts);

plan = PriorityPlanner.buildPlan(mapData, numUAV, struct());

pos = plan.positions;                 % [T x U x 2] (row,col)
disp(size(pos));

H        = mapData.H;
Z_m      = mapData.Z_m;
obstacle = double(mapData.obstacle);
W        = mapData.W;
starts   = plan.starts;

save('-v7', fullfile(projRoot, 'drone_field_export.mat'), ...
    'H', 'Z_m', 'obstacle', 'W', 'pos', 'starts', 'dx', 'N', 'M');

fprintf('Export complete: %d timesteps, %d UAV(s).\n', size(pos,1), size(pos,2));
