function mu_shuff = spatiotemporal_shuffle_complex_mu(mu_trials)
% SPATIOTEMPORAL_SHUFFLE_COMPLEX_MU
%
% Input:
%   mu_trials : nTrials x 1 cell array
%               each cell = [time x channels] complex mu analytic signal
%
% Output:
%   mu_shuff  : same structure and same trial lengths
%
% For each trial:
%   1) randomly permute time points
%   2) randomly permute channel identities
%
% This preserves:
%   - trial length
%   - marginal distribution of complex mu values
%
% This destroys:
%   - temporal continuity
%   - true spatial phase organization

nTrials = numel(mu_trials);
mu_shuff = cell(size(mu_trials));

for tr = 1:nTrials

    X = mu_trials{tr};   % [time x channels]

    nT  = size(X,1);
    nCh = size(X,2);

    % Temporal shuffle
    t_idx = randperm(nT);

    % Spatial shuffle
    ch_idx = randperm(nCh);

    % Apply both
    mu_shuff{tr} = X(t_idx, ch_idx);

end

end