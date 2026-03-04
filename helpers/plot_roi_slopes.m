function outTbl=plot_roi_slopes(T,lme_roi)


% === ROI-specific Day slopes from LME + bar plot with 95% CI + significance stars ===
% Assumes:
%   lme_roi is fit with:  'Y ~ 1 + DayC*ROI + (1|ChanID)'
%   T.ROI exists (categorical)
%   DayC is centered (doesn't matter for Day-slope extraction)

roiLevels = categories(T.ROI);
nROI = numel(roiLevels);

b     = fixedEffects(lme_roi);
names = lme_roi.CoefficientNames;
V     = lme_roi.CoefficientCovariance;
df    = lme_roi.DFE;   % approx df for fixed effects

% Base day slope for reference ROI
idxDay = find(strcmp(names,'DayC'));
bDay   = b(idxDay);

roiSlope   = zeros(nROI,1);
roiSE      = zeros(nROI,1);
roiP       = zeros(nROI,1);
roiCI95_lo = zeros(nROI,1);
roiCI95_hi = zeros(nROI,1);

tcrit = tinv(0.975, df);

for r = 1:nROI
    roiName = roiLevels{r};

    % Non-reference levels have an interaction coefficient DayC:ROI_<level>
    term = ['DayC:ROI_' roiName];
    idxInt = find(strcmp(names, term));

    if isempty(idxInt)
        % reference ROI
        roiSlope(r) = bDay;
        roiSE(r)    = sqrt(V(idxDay, idxDay));
    else
        roiSlope(r) = bDay + b(idxInt);
        roiSE(r)    = sqrt(V(idxDay, idxDay) + V(idxInt, idxInt) + 2*V(idxDay, idxInt));
    end

    tval = roiSlope(r) / roiSE(r);
    roiP(r) = 2 * tcdf(-abs(tval), df);

    roiCI95_lo(r) = roiSlope(r) - tcrit*roiSE(r);
    roiCI95_hi(r) = roiSlope(r) + tcrit*roiSE(r);
end

% get t-value
df = lme_roi.DFE;
tval = roiSlope ./ roiSE;

outTbl = table(string(roiLevels), roiSlope, roiSE, tval, roiCI95_lo, roiCI95_hi, roiP, ...
               repmat(df, numel(roiLevels), 1), ...
    'VariableNames', {'ROI','DaySlope','SE','t','CI95_L','CI95_U','p','DF'});

disp(outTbl)

% ---- Table output (optional) ----
% outTbl = table(string(roiLevels), roiSlope, roiSE, roiCI95_lo, roiCI95_hi, roiP, ...
%     'VariableNames', {'ROI','DaySlope','SE','CI95_L','CI95_U','p'});
% disp(outTbl)

% ---- Bar plot with 95% CI and stars ----
figure; hold on
x = 1:nROI;

bar(roiSlope);
yline(0,'-k');

% CI errorbars
errLo = roiSlope - roiCI95_lo;
errHi = roiCI95_hi - roiSlope;
errorbar(x, roiSlope, errLo, errHi, 'k', 'linestyle','none', 'LineWidth', 1.2);

xticks(x); xticklabels(roiLevels); xtickangle(45);
ylabel('Slope of change in relative decoding across-days');
box on

% Stars (uncorrected p<0.05,0.01,0.001)
yl = ylim; yr = yl(2) - yl(1);
pad = 0.03 * yr;

for i = 1:nROI
    if roiP(i) < 0.05
        if roiP(i) < 0.001
            s = '***';
        elseif roiP(i) < 0.01
            s = '**';
        else
            s = '*';
        end

        % place star above upper CI if positive, below lower CI if negative
        if roiSlope(i) >= 0
            yStar = roiCI95_hi(i) + pad;
            va = 'bottom';
        else
            yStar = roiCI95_lo(i) - pad;
            va = 'top';
        end

        text(x(i), yStar, s, 'HorizontalAlignment','center', ...
            'VerticalAlignment', va, 'FontSize', 12, 'FontWeight','bold');
    end
end

% expand ylim a bit so stars don't clip
ylim([yl(1)-0.08*yr, yl(2)+0.12*yr]);