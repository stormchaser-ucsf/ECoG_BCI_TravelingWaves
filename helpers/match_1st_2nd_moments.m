function Xw_aligned = match_1st_2nd_moments(Xw, Xnw, nt_w, nt_nw)

    D = size(Xw,2);

    if length(nt_w) ~= length(nt_nw)
        error('Wave and nonwave must have same number of trials for trial-wise mean replacement.');
    end

    %% ===== 1) Demean EACH TRIAL (remove between covariance) =====
    [Xw_within, mu_w_trials]  = demean_trials(Xw, nt_w);
    [Xnw_within, mu_nw_trials] = demean_trials(Xnw, nt_nw);

    %% ===== 2) Compute WITHIN covariances =====
    Sigma_within_w  = cov(Xw_within);
    Sigma_within_nw = cov(Xnw_within);

    %% ===== 3) Regularization (important in high-D) =====
    eps_w  = 1e-6 * trace(Sigma_within_w)/D;
    eps_nw = 1e-6 * trace(Sigma_within_nw)/D;

    Sigma_within_w  = Sigma_within_w  + eps_w  * eye(D);
    Sigma_within_nw = Sigma_within_nw + eps_nw * eye(D);

    %% ===== 4) Whitening–Recoloring (WITHIN only) =====
    [Uw, Sw]   = eig(Sigma_within_w);
    [Unw, Snw] = eig(Sigma_within_nw);

    Sw_inv_sqrt = Uw  * diag(1 ./ sqrt(diag(Sw))) * Uw';
    Snw_sqrt    = Unw * diag(sqrt(diag(Snw)))     * Unw';

    A = Snw_sqrt * Sw_inv_sqrt;

    %% ===== 5) Apply transform to WITHIN-demeaned wave =====
    Xw_transformed = (A * Xw_within')';

    %% ===== 6) Add back NONWAVE trial means (restores between covariance) =====
    Xw_aligned = zeros(size(Xw));
    start_idx = 1;

    for t = 1:length(nt_w)

        n = nt_w(t);
        idx = start_idx:(start_idx+n-1);

        % Add nonwave trial mean
        Xw_aligned(idx,:) = Xw_transformed(idx,:) + mu_nw_trials(t,:);

        start_idx = start_idx + n;
    end

end


%% ===== Helper: Demean each trial =====
function [X_demeaned, mu_trials] = demean_trials(X, trial_lengths)

    X_demeaned = zeros(size(X));
    mu_trials  = zeros(length(trial_lengths), size(X,2));

    start_idx = 1;

    for t = 1:length(trial_lengths)

        n = trial_lengths(t);
        idx = start_idx:(start_idx+n-1);

        Xt = X(idx,:);
        mu_t = mean(Xt,1);

        X_demeaned(idx,:) = Xt - mu_t;
        mu_trials(t,:) = mu_t;

        start_idx = start_idx + n;
    end

end



%% function Xw_aligned = match_1st_2nd_moments(Xw, Xnw, nt_w, nt_nw)
% 
%     D = size(Xw,2);
% 
%     %% ===== 1) Means =====
%     mu_w  = mean(Xw,1);
%     mu_nw = mean(Xnw,1);
% 
%     %% ===== 2) TOTAL covariance =====
%     Sigma_total_w  = cov(Xw);
%     Sigma_total_nw = cov(Xnw);
% 
%     %% ===== 3) WITHIN covariance =====
%     Sigma_within_w  = compute_within_cov(Xw, nt_w);
%     Sigma_within_nw = compute_within_cov(Xnw, nt_nw);
% 
%     %% ===== 4) BETWEEN covariance =====
%     Sigma_between_w  = Sigma_total_w  - Sigma_within_w;
%     Sigma_between_nw = Sigma_total_nw - Sigma_within_nw;
% 
%     % Numerical cleanup
%     Sigma_between_w  = (Sigma_between_w  + Sigma_between_w')/2;
%     Sigma_between_nw = (Sigma_between_nw + Sigma_between_nw')/2;
% 
%     %% ===== 5) Use TOTAL covariance for matching =====
%     Sigma_w  = Sigma_total_w;
%     Sigma_nw = Sigma_total_nw;
% 
%     %% ===== 6) Regularization (important in 253D) =====
%     eps_w  = 1e-6 * trace(Sigma_w)/D;
%     eps_nw = 1e-6 * trace(Sigma_nw)/D;
% 
%     Sigma_w  = Sigma_w  + eps_w  * eye(D);
%     Sigma_nw = Sigma_nw + eps_nw * eye(D);
% 
%     %% ===== 7) Whitening–Recoloring Transform =====
%     [Uw, Sw]   = eig(Sigma_w);
%     [Unw, Snw] = eig(Sigma_nw);
% 
%     Sw_inv_sqrt = Uw  * diag(1 ./ sqrt(diag(Sw))) * Uw';
%     Snw_sqrt    = Unw * diag(sqrt(diag(Snw)))     * Unw';
% 
%     A = Snw_sqrt * Sw_inv_sqrt;
% 
%     %% ===== 8) Apply affine transform =====
%     Xc = Xw - mu_w;
%     Xc = (A * Xc')';
%     Xw_aligned = Xc + mu_nw;
% 
% end
% 
% 
% %% ===== Helper: Within-trial covariance =====
% function Sigma_within = compute_within_cov(X, trial_lengths)
% 
%     X_demeaned = zeros(size(X));
%     start_idx = 1;
% 
%     for t = 1:length(trial_lengths)
% 
%         n = trial_lengths(t);
%         idx = start_idx:(start_idx+n-1);
% 
%         Xt = X(idx,:);
%         Xt = Xt - mean(Xt,1);
% 
%         X_demeaned(idx,:) = Xt;
% 
%         start_idx = start_idx + n;
%     end
% 
%     Sigma_within = cov(X_demeaned);
% 
% end