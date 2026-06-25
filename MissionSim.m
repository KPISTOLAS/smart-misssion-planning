classdef MissionSim
    % MISSIONSIM Multi-UAV execution engine with 3D-ish power split (horizontal / vertical / blade profile),
    %   optional fleet energy budget (battery sortie cap), dynamic IoT drift, information gain in bits (EIG proxy),
    %   collision monitoring, and coverage bookkeeping.

    methods (Static)

        function log = run(mapData, plan, cfg)
            arguments
                mapData struct
                plan struct
                cfg struct = struct()
            end

            cfg = MissionSim.defaultCfg(cfg);

            if ~isfield(mapData, 'eig_bits')
                mapDataLive = GPInformationField.attach(mapData, struct());
            else
                mapDataLive = mapData;
            end

            positions = plan.positions;
            [Tplan, U, ~] = size(positions);

            dx = mapDataLive.dx;
            trav = ~mapDataLive.obstacle;
            Z = mapDataLive.Z_m;

            Hloc = double(mapDataLive.H);
            sigloc = double(mapDataLive.sigma);

            visited = false(mapDataLive.N, mapDataLive.M);
            igAccum = 0;
            bitsAccum = 0;

            posHist = zeros(Tplan, U, 2);
            spdHist = zeros(Tplan, U);
            altHist = zeros(Tplan, U);
            vzHist = zeros(Tplan, U);
            enHist = zeros(Tplan, 1);
            colHist = false(Tplan, 1);

            cumArea = zeros(Tplan, 1);
            energy = 0;

            tStop = Tplan;

            for t = 1:Tplan
                if cfg.dynamicIoTEnable
                    maskFree = ~mapDataLive.obstacle;
                    Hloc = Hloc + cfg.dynamicSigmaH .* randn(size(Hloc)) .* double(maskFree);
                    Hloc = min(4, max(0, round(Hloc)));
                    sigloc = sigloc + cfg.dynamicSigmaS .* randn(size(sigloc)) .* double(maskFree);
                    sigloc = min(1, max(0.05, sigloc));

                    % Refresh GP-derived bits map after uncertainty drift.
                    tmp = mapDataLive;
                    tmp.H = Hloc;
                    tmp.sigma = sigloc;
                    tmp.highPriorityMask = (Hloc >= cfg.highHealthThr);
                    tmp = GPInformationField.attach(tmp, struct('sigmaMeasNoise', cfg.sigmaMeasNoise));
                    eigBitsNow = tmp.eig_bits;
                else
                    eigBitsNow = mapDataLive.eig_bits;
                end

                highMask = (Hloc >= cfg.highHealthThr) & trav;

                posHist(t, :, :) = positions(t, :, :);

                if t > 1
                    for u = 1:U
                        r0 = round(positions(t - 1, u, 1));
                        c0 = round(positions(t - 1, u, 2));
                        r = round(positions(t, u, 1));
                        c = round(positions(t, u, 2));

                        r0 = max(1, min(mapDataLive.N, r0));
                        c0 = max(1, min(mapDataLive.M, c0));
                        r = max(1, min(mapDataLive.N, r));
                        c = max(1, min(mapDataLive.M, c));

                        dr = r - r0;
                        dc = c - c0;
                        stepM = hypot(dr, dc) * dx;

                        if stepM > cfg.stepTolM + eps
                            stepM = min(stepM, cfg.vMax * cfg.dt);
                        end

                        dz = Z(r, c) - Z(r0, c0);
                        vz = dz / max(cfg.dt, eps);
                        vh = stepM / max(cfg.dt, eps);

                        agl = cfg.altitudeAGL + Z(r, c);
                        spdHist(t, u) = vh;
                        vzHist(t, u) = vz;
                        altHist(t, u) = agl;

                        if cfg.usePowerModel3D
                            P = MissionSim.powerTotal3D(vh, vz, agl, cfg);
                        else
                            P = MissionSim.powerWattsLegacy(vh, agl, cfg);
                        end

                        energy = energy + P * cfg.dt;
                    end
                else
                    for u = 1:U
                        r = round(positions(t, u, 1));
                        c = round(positions(t, u, 2));
                        r = max(1, min(mapDataLive.N, r));
                        c = max(1, min(mapDataLive.M, c));
                        altHist(t, u) = cfg.altitudeAGL + Z(r, c);
                        spdHist(t, u) = 0;
                        vzHist(t, u) = 0;

                        if cfg.usePowerModel3D
                            P = MissionSim.powerTotal3D(0, 0, altHist(t, u), cfg);
                        else
                            P = MissionSim.powerWattsLegacy(0, altHist(t, u), cfg);
                        end

                        energy = energy + P * cfg.dt;
                    end
                end

                colHist(t) = MissionSim.pairwiseViolation(positions, t, U, dx, cfg.dSafe);

                for u = 1:U
                    r = round(positions(t, u, 1));
                    c = round(positions(t, u, 2));

                    if r < 1 || r > mapDataLive.N || c < 1 || c > mapDataLive.M
                        continue;
                    end

                    if ~trav(r, c)
                        continue;
                    end

                    if ~visited(r, c)
                        igAccum = igAccum + sigloc(r, c);
                        bitsAccum = bitsAccum + eigBitsNow(r, c);
                    end

                    visited(r, c) = true;
                end

                cumArea(t) = nnz(visited & trav) * mapDataLive.cellArea_m2;
                enHist(t) = energy;

                hiCells = nnz(highMask & trav);

                if hiCells > 0
                    hpcNow = 100 * nnz(visited & highMask) / hiCells;
                else
                    hpcNow = 0;
                end

                if ~isempty(cfg.stopHpcPct) && hpcNow >= cfg.stopHpcPct
                    tStop = t;
                    break;
                end

                if ~isempty(cfg.fleetEnergyBudget_J) && energy >= cfg.fleetEnergyBudget_J
                    tStop = t;
                    break;
                end
            end

            if tStop < Tplan
                posHist = posHist(1:tStop, :, :);
                spdHist = spdHist(1:tStop, :, :);
                altHist = altHist(1:tStop, :, :);
                vzHist = vzHist(1:tStop, :, :);
                enHist = enHist(1:tStop);
                colHist = colHist(1:tStop);
                cumArea = cumArea(1:tStop);
            end

            log = struct();
            log.positions = posHist;
            log.speed_mps = spdHist;
            log.altitude_m = altHist;
            log.vertical_speed_mps = vzHist;
            log.energy_J = enHist;
            log.collision = colHist;
            log.coveredArea_m2 = cumArea;
            log.visited = visited;
            log.totalEnergy_J = energy;
            log.totalInformationGain = igAccum;
            log.total_bits_information = bitsAccum;
            log.planName = plan.name;
            log.dt = cfg.dt;
            log.highMask = highMask;
            log.truncatedSteps = tStop < Tplan;
        end

        function cfg = defaultCfg(cfg)
            d = struct( ...
                'dt', 1.0, ...
                'vMax', 12, ...
                'dSafe', 25, ...
                'altitudeAGL', 35, ...
                'P0', 180, ...
                'kSpeed', 0.35, ...
                'kAlt', 0.85, ...
                'stepTolM', 25, ...
                'usePowerModel3D', true, ...
                'kHorizParasite', 0.22, ...
                'kClimb', 14.5, ...
                'kDescent', 2.1, ...
                'kBladeProfile', 0.0018, ...
                'fleetEnergyBudget_J', [], ...
                'stopHpcPct', [], ...
                'dynamicIoTEnable', false, ...
                'dynamicSigmaH', 0.08, ...
                'dynamicSigmaS', 0.015, ...
                'sigmaMeasNoise', 0.12, ...
                'highHealthThr', 3);

            fn = fieldnames(d);
            for k = 1:numel(fn)
                if ~isfield(cfg, fn{k}) || isempty(cfg.(fn{k}))
                    cfg.(fn{k}) = d.(fn{k});
                end
            end
        end

        function P = powerTotal3D(v_horiz, vz, altitude_m, cfg)
            % PTOTAL3D — blade/induced baseline + horizontal parasite/induced + asymmetric climb/descent.

            P_blade = cfg.P0 * (1 + cfg.kBladeProfile * (v_horiz .^ 2));
            P_horiz = cfg.kSpeed * (abs(v_horiz) .^ 3) + cfg.kHorizParasite * (v_horiz .^ 2);
            P_vert = cfg.kClimb * (max(0, vz) .^ 2) + cfg.kDescent * (max(0, -vz) .^ 2);
            P_alt = cfg.kAlt * altitude_m;
            P = P_blade + P_horiz + P_vert + P_alt;
            P = max(P, 0.35 * cfg.P0);
        end

        function P = powerWattsLegacy(v, altitude_m, cfg)
            P = cfg.P0 + cfg.kSpeed * (v .^ 3) + cfg.kAlt * altitude_m;
            P = max(P, 0.4 * cfg.P0);
        end

        function flag = pairwiseViolation(positions, t, U, dx, dSafe)
            flag = false;

            for a = 1:U
                for b = a + 1:U
                    pa = squeeze(positions(t, a, :));
                    pb = squeeze(positions(t, b, :));
                    dist = hypot((pa(1) - pb(1)) * dx, (pa(2) - pb(2)) * dx);

                    if dist < dSafe
                        flag = true;
                        return;
                    end
                end
            end
        end
    end
end
