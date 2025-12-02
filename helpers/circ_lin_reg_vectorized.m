function result = circ_lin_reg_vectorized(x, y, phase, rho_grid, alpha_grid_deg)
% Fully vectorized 2D circular-linear regression
% Fits: phase ≈ a*x + b*y + theta0 (mod 2π)


alpha_grid = deg2rad(alpha_grid_deg);

% Ensure column vectors
x = x(:); 
y = y(:); 
phase = wrapTo2Pi(phase(:));

N = numel(x);
Nrho = numel(rho_grid);
Nalpha = numel(alpha_grid);

% Create all combinations of (rho, alpha)
[RHO, ALPHA] = ndgrid(rho_grid, alpha_grid);

% Convert polar params to slopes a, b
A = RHO .* cos(ALPHA);   % size [Nrho,Nalpha]
B = RHO .* sin(ALPHA);   % size [Nrho,Nalpha]

% Expand x, y across parameter grid
% x: [N,1] → [N, 1, 1]
x3 = reshape(x, [N, 1, 1]);
y3 = reshape(y, [N, 1, 1]);

% A, B: [Nrho,Nalpha] → [1, Nrho, Nalpha]
A3 = reshape(A, [1, Nrho, Nalpha]);
B3 = reshape(B, [1, Nrho, Nalpha]);

% Predicted phase before offset: ax + by
pred = x3 .* A3 + y3 .* B3;  % size [N, Nrho, Nalpha]
pred = wrapTo2Pi(pred);

% Compute best circular offset θ0 for each (rho,alpha)
phase3 = wrapTo2Pi(reshape(phase, [N,1,1]));
complex_diff = exp(1i * (phase3 - pred));
theta0 = angle(sum(complex_diff, 1));   % size [1,Nrho,Nalpha]

% Apply offset
pred_final = wrapTo2Pi(pred + theta0);

% Compute circular correlation (vectorized)
theta_bar = angle(mean(exp(1i*pred_final),1));      % [1,Nrho,Nalpha]
phi_bar   = angle(mean(exp(1i*phase3),1));          % scalar → auto-expanded

num = sum( sin(pred_final - theta_bar) .* sin(phase3 - phi_bar), 1 );
den = sqrt( sum(sin(pred_final - theta_bar).^2,1) .* sum(sin(phase3 - phi_bar).^2,1) );

R = num ./ den;  % size [1,Nrho,Nalpha]

% Find best-fit parameters
[R_best, idx] = max(R(:));
[idx_rho, idx_alpha] = ind2sub(size(R), idx);

% Pack results
result.rho   = rho_grid(idx_rho);
result.alpha = alpha_grid(idx_alpha);
result.a     = result.rho * cos(result.alpha);
result.b     = result.rho * sin(result.alpha);
result.theta0 = theta0(1,idx_rho,idx_alpha);
result.R      = R_best;
end
