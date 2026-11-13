function mu_shuff = temporal_shuffle_all_trials_no_replace(mu_trials)
% TEMPORAL_SHUFFLE_ALL_TRIALS_NO_REPLACE
%
% Input:
%   mu_trials : nTrials x 1 cell array
%               each cell = [time x channels] complex mu data
%
% Output:
%   mu_shuff  : same cell structure and same trial lengths
%
% Procedure:
%   1) Pool every time point from every trial
%   2) Randomly permute ALL pooled time points WITHOUT replacement
%   3) Keep all channels together at each time point
%   4) Repartition into surrogate trials with the ORIGINAL trial lengths
%
% Preserves:
%   - every original multichannel spatial snapshot exactly once
%   - instantaneous spatial phase structure
%   - total number of samples
%   - original trial-length distribution
%
% Destroys:
%   - temporal continuity
%   - original trial identity / temporal ordering

nTrials = numel(mu_trials);

% Number of channels
nCh = size(mu_trials{1}, 2);

% Original trial lengths
trial_lengths = zeros(nTrials,1);

for tr = 1:nTrials
    trial_lengths(tr) = size(mu_trials{tr},1);

    if size(mu_trials{tr},2) ~= nCh
        error('All trials must have the same number of channels.');
    end
end

% ------------------------------------------------------------
% Pool all temporal samples across trials
% pooled_data = total_timepoints x channels
% ------------------------------------------------------------
pooled_data = vertcat(mu_trials{:});

nTotal = size(pooled_data,1);

% ------------------------------------------------------------
% Shuffle ALL time points WITHOUT replacement
% Spatial snapshot remains intact
% ------------------------------------------------------------
idx = randperm(nTotal);

pooled_shuff = pooled_data(idx,:);

% ------------------------------------------------------------
% Reconstruct surrogate trials with original lengths
% ------------------------------------------------------------
mu_shuff = cell(size(mu_trials));

start_idx = 1;

for tr = 1:nTrials

    stop_idx = start_idx + trial_lengths(tr) - 1;

    mu_shuff{tr} = pooled_shuff(start_idx:stop_idx,:);

    start_idx = stop_idx + 1;

end
end