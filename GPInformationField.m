classdef GPInformationField
    % GPINFORMATIONFIELD GP-style marginal variance and Expected Information Gain (EIG) proxy.
    %   Interprets IoT telemetry sigma as posterior marginal std on a latent health field.
    %   Scalar Gaussian measurement noise sigma_n yields approximate entropy reduction (bits)
    %   before visiting a cell; active sensing prioritizes high variance among high-stress sites.
    %
    %   See also PriorityPlanner, MapGenerator.

    methods (Static)

        function mapData = attach(mapData, opts)
            arguments
                mapData struct
                opts struct = struct()
            end

            if ~isfield(opts, 'sigmaMeasNoise') || isempty(opts.sigmaMeasNoise)
                opts.sigmaMeasNoise = 0.12;
            end

            sigma_n = opts.sigmaMeasNoise;
            v = mapData.sigma.^2;
            mapData.gp_variance_marginal = v;

            % Expected entropy reduction for noisy Gaussian observation (bits, natural log base 2).
            mapData.eig_bits = 0.5 * log2(1 + v / max(eps, sigma_n^2));

            trav = ~mapData.obstacle;
            hi = mapData.highPriorityMask & trav;

            % Active sensing score: EIG restricted to high-stress actionable region.
            mapData.activeSensingScore = mapData.eig_bits .* double(hi);

            % Differentiable surrogate for visualization / blending (not normalized).
            mapData.mutual_information_proxy = mapData.eig_bits .* (1 + 0.25 * double(mapData.H));

            % Store measurement noise used (for documentation).
            mapData.infoMeta = opts;
        end
    end
end
