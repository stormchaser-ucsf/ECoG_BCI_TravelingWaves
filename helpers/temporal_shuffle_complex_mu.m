function mu_shuff = temporal_shuffle_complex_mu(mu_trials)
% TEMPORAL_SHUFFLE_COMPLEX_MU
%
% Input:
%   mu_trials : nTrials x 1 cell array
%               each cell = [time x channels] complex mu analytic signal
%
% Output:
%   mu_shuff  : same structure and same trial lengths
%
% For each trial, time points are randomly permuted while the entire
% multichannel spatial snapshot is kept intact.
%
% This preserves:
%   - trial length
%   - instantaneous spatial phase/amplitude pattern
%   - distribution of complex mu values within each trial
%
% This destroys:
%   - temporal continuity of phase-gradient structure
%   - sustained traveling-wave epochs

nTrials = numel(mu_trials);
mu_shuff = cell(size(mu_trials));

for tr = 1:nTrials

    X = mu_trials{tr};   % [time x channels]

    nT = size(X,1);

    % Random permutation of time points
    idx = randperm(nT);

    % Move the entire multichannel snapshot together
    mu_shuff{tr} = X(idx,:);

end

end