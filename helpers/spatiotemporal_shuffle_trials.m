function hold_mu_shuff = spatiotemporal_shuffle_trials(hold_mu)
% SPATIOTEMPORAL_SHUFFLE_TRIALS
%
% Input:
%   hold_mu : nTrials x 1 cell array
%             each cell = [time x channels]
%
% Output:
%   hold_mu_shuff : nTrials x 1 cell array
%                   same trial lengths and number of channels as input
%
% Null construction:
%   At each time point:
%       1. Find all trials that contain that time point.
%       2. Randomly permute trial identity across those trials.
%       3. Copy the entire channel snapshot from the shuffled source trial.
%       4. Randomly permute channel identities for that snapshot.
%
% This destroys:
%   - temporal continuity within trials
%   - spatial organization across electrodes
%
% while preserving:
%   - original trial lengths
%   - instantaneous amplitude/value distribution
%   - number of channels

nTrials = numel(hold_mu);

% Check number of channels is consistent
nCh = size(hold_mu{1}, 2);

trial_lengths = zeros(nTrials,1);

for tr = 1:nTrials
    trial_lengths(tr) = size(hold_mu{tr},1);

    if size(hold_mu{tr},2) ~= nCh
        error('All trials must have the same number of channels.');
    end
end

maxT = max(trial_lengths);

% Preallocate output with same dimensions as original trials
hold_mu_shuff = cell(size(hold_mu));

for tr = 1:nTrials
    hold_mu_shuff{tr} = zeros(size(hold_mu{tr}), 'like', hold_mu{tr});
end


% -------------------------------------------------------------
% Shuffle independently at every time point
% -------------------------------------------------------------
for t = 1:maxT

    % Trials that actually contain this time point
    validTrials = find(trial_lengths >= t);

    nValid = numel(validTrials);

    if nValid < 2
        % Nothing meaningful to shuffle
        continue
    end

    % Shuffle trial labels WITHOUT replacement
    sourceTrials = validTrials(randperm(nValid));


    % Assign shuffled snapshots to surrogate trials
    for ii = 1:nValid

        targetTrial = validTrials(ii);
        sourceTrial = sourceTrials(ii);

        % Entire multichannel snapshot at time t
        snapshot = hold_mu{sourceTrial}(t,:);

        % Spatial shuffle: destroy electrode topology
        ch_perm = randperm(nCh);

        snapshot = snapshot(ch_perm);

        % Put into shuffled trial
        hold_mu_shuff{targetTrial}(t,:) = snapshot;

    end
end

end