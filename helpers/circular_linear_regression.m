function [result] = circular_linear_regression(x, y, phase, rho_grid, alpha_grid)
% circular_linear_regression
% ---------------------------------------------------------
% Fits phase(x,y) ≈ a*x + b*y + theta0 (mod 2π)
% via grid search on:
%   rho   = spatial frequency (phase slope magnitude)
%   alpha = direction of propagation
%
% INPUTS:
%   x, y    : electrode coordinates (vectors, same length)
%   phase   : phase in radians (vector, same length)
%   rho_grid: vector of rho values to search (e.g., linspace(0, rho_max, 50))
%   alpha_grid: vector of alpha values (e.g., linspace(0, 2*pi, 180))
%
% OUTPUT:
%   result.rho
%   result.alpha
%   result.a
%   result.b
%   result.theta0
%   result.R (circular correlation / goodness-of-fit)

phase = wrapTo2Pi(phase(:));
x = x(:); y = y(:);

best_R = -inf;
best_params = struct();

% Loop over grid
for ri = 1:length(rho_grid)
    rho = rho_grid(ri);

    for ai = 1:length(alpha_grid)
        alpha = alpha_grid(ai);

        % Convert polar parameters to slopes
        a = rho * cos(alpha);
        b = rho * sin(alpha);

        % Predicted phase before wrapping
        pred = a*x + b*y;
        pred = wrapTo2Pi(pred);

        % Fit circular offset θ0 analytically:
        % minimize angle difference by aligning mean directions
        complex_diff = exp(1i*(phase - pred));
        theta0 = angle(sum(complex_diff));

        pred_final = wrapTo2Pi(pred + theta0);

        % Compute circular correlation
        R = circ_corrcc(pred_final(:), phase(:));

        % Track best
        if R > best_R
            best_R = R;
            best_params = struct('rho', rho, 'alpha', alpha, ...
                                 'a', a, 'b', b, 'theta0', theta0, ...
                                 'R', R);
        end
    end
end

result = best_params;
end
