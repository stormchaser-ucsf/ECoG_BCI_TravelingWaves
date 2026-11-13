function shuffled_trials = full_spatiotemporal_shuffle(trials)
% FULL_SPATIOTEMPORAL_SHUFFLE
%
% Input:
%   trials : nTrials x 1 cell array
%            each cell is [time x channels]
%            trial lengths can differ
%
% Output:
%   shuffled_trials : same cell structure and same trial lengths
%
% Null destroys:
%   1) temporal continuity within trials
%   2) trial identity / event-relative timing
%   3) spatial organization across electrodes
%
% Preserves:
%   1) total number of samples
%   2) original trial-length distribution
%   3) instantaneous multichannel values before spatial permutation
%   4) marginal distribution of signal values

nTrials = numel(trials);

% ------------------------------------------------------------
% Basic checks + original trial lengths
% ------------------------------------------------------------
nCh = size(trials{1}, 2);
trial_lengths = zeros(nTrials,1);

for tr = 1:nTrials
    trial_lengths(tr) = size(trials{tr},1);

    if size(trials{tr},2) ~= nCh
        error('All trials must contain the same number of channels.');
    end
end

% ------------------------------------------------------------
% Concatenate ALL time samples across ALL trials
%
% pooled_data:
%      total_timepoints x channels
% ------------------------------------------------------------
pooled_data = vertcat(trials{:});

nTotal = size(pooled_data,1);

% ------------------------------------------------------------
% TEMPORAL SHUFFLE
%
% Every multichannel snapshot can now come from ANY time point
% in ANY original trial.
% ------------------------------------------------------------
time_perm = randperm(nTotal);

pooled_shuff = pooled_data(time_perm,:);

% ------------------------------------------------------------
% SPATIAL SHUFFLE
%
% Independently permute electrode locations at each time point.
% This destroys coherent spatial phase gradients.
% ------------------------------------------------------------
for t = 1:nTotal

    ch_perm = randperm(nCh);

    pooled_shuff(t,:) = pooled_shuff(t,ch_perm);

end

% ------------------------------------------------------------
% Reconstruct surrogate trials using EXACT original lengths
% ------------------------------------------------------------
shuffled_trials = cell(size(trials));

idx_start = 1;

for tr = 1:nTrials

    idx_end = idx_start + trial_lengths(tr) - 1;

    shuffled_trials{tr} = pooled_shuff(idx_start:idx_end,:);

    idx_start = idx_end + 1;

end

end