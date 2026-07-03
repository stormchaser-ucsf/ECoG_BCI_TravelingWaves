%% init


clear
clc
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))
cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves')


%% LOAD IMAGING DATA FOR THE NEURON DATASET

imaging_EC189

imaging_EC210

imaging_EC176

%% Get hG and LFO ERPs example


%% PAC between hG and mu


%% PAC between hG and LFO

cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC176_ProcessingForNikhilesh/ecog_data_NN')
ec176 = load('EC176_sig_ch_LFO_hG_PAC.mat');

cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210')
ec210 = load('sig_ch_LFO_hG_PAC.mat');

cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189')
ec189 = load('EC189_sig_ch_LFO_hG_PAC.mat');


% ec210 image of pac lfo hg on brain

% plot on brain
% hold
bad_chI = ec210.bad_chI;
plv_hold = ec210.plv_hold;
plv_move = ec210.plv_move;

a = exp(1i*plv_hold);
a = mean(a,1);
a = abs(a);

b = exp(1i*plv_move);
b = mean(b,1);
b = abs(b);

% sig testing
pval_hold=[];pval_move=[];
for i=1:size(plv_hold,2)
    if bad_chI(i)==0
        pval_hold(i)=NaN;
        pval_move(i)=NaN;
    else
        ang = plv_hold(:,i);
        [p,z]=circ_rtest(ang);
        pval_hold(i) = p;

        ang = plv_move(:,i);
        [p,z]=circ_rtest(ang);
        pval_move(i) = p;
    end
end
[pfdrh,pval] =fdr(pval_hold(~isnan(pval_hold)),0.05);
sum(pval_hold<=pfdrh)
sig_ch_hold = (pval_hold<=pfdrh);
[pfdrm,pval] =fdr(pval_move(~isnan(pval_move)),0.05);
sum(pval_move<=pfdrm)
sig_ch_move = (pval_move<=pfdrm);

figure;
val=a;
good_ch = find(bad_chI==1);
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'rh',1,1,1);
e_h = el_add(elecmatrix([good_ch],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*12;
    c='b';
    if ms>0.0 && bad_chI(j)==1 && sig_ch_hold(j)==1
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end
plot_beautify
% move
figure;
val=b;
good_ch = find(bad_chI==1);
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'rh',1,1,1);
e_h = el_add(elecmatrix([good_ch],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*12;
    c='b';
    if ms>0.0 && bad_chI(j)==1 && sig_ch_move(j)==1
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end
plot_beautify


% plot now percent channels significant
subj_names = {'ec176','ec189','ec210'};
subj.ec176 = ec176;
subj.ec189 = ec189;
subj.ec210 = ec210;
hold_per_sig=[];
move_per_sig=[];
plv_move_all=[];
plv_hold_all=[];
idxx=[];
for i=1:length(subj_names)

    bad_chI = subj.(subj_names{i}).bad_chI;
    plv_hold = subj.(subj_names{i}).plv_hold;
    plv_move = subj.(subj_names{i}).plv_move;
    good_ch = find(bad_chI==1);

    a = exp(1i*plv_hold);
    a = mean(a,1);
    a = abs(a);
    plv_hold_all =[plv_hold_all; a(good_ch)'];
    

    b = exp(1i*plv_move);
    b = mean(b,1);
    b = abs(b);
    plv_move_all =[plv_move_all; b(good_ch)'];
    idxx = [idxx;i*ones(length(good_ch),1)];

    % sig testing
    pval_hold=[];pval_move=[];
    for ii=1:size(plv_hold,2)
        if bad_chI(ii)==0
            pval_hold(ii)=NaN;
            pval_move(ii)=NaN;
        else
            ang = plv_hold(:,ii);
            [p,z]=circ_rtest(ang);
            pval_hold(ii) = p;

            ang = plv_move(:,ii);
            [p,z]=circ_rtest(ang);
            pval_move(ii) = p;
        end
    end
    [pfdrh,pval] =fdr(pval_hold(~isnan(pval_hold)),0.05);
    sum(pval_hold<=pfdrh);
    sig_ch_hold = (pval_hold<=pfdrh);
    [pfdrm,pval] =fdr(pval_move(~isnan(pval_move)),0.05);
    sum(pval_move<=pfdrm);
    sig_ch_move = (pval_move<=pfdrm);

    hold_per_sig(i) =  sum(pval_hold(~isnan(pval_hold)) <= pfdrh)/sum(bad_chI);
    move_per_sig(i) =  sum(pval_move(~isnan(pval_move)) <= pfdrm)/sum(bad_chI);
end


d = [hold_per_sig' move_per_sig'];
figure;
bar([1 2],mean(d), 'BarWidth', 0.5,'FaceColor',[.75 .75 .75 ])
hold on
idx = ones(size(d,1),1) + 0.1*randn(size(d,1),1);
plot(idx,d(:,1),'.b','MarkerSize',40)
idx = 2*ones(size(d,1),1) + 0.1*randn(size(d,1),1);
plot(idx,d(:,2),'.b','MarkerSize',40)
xticks(1:2)
xticklabels({'Move','Hold'})
yticks(0:.05:1)
xlim([.5 2.5])
plot_beautify

% now scatter plv values each subject
for i=1:3
    figure;
    idx=find(idxx==i);
    d=[plv_hold_all(idx) plv_move_all(idx)];
    boxplot(d)   
    xticks(1:2)
    xticklabels({'Move','Hold'})
    plot_beautify
    title(subj_names{i})
    ylim([0 0.9])
    yticks([0:.1:.8])
end

%all at once
plv_hold_all = plv_hold_all(:);
plv_move_all = plv_move_all(:);
idxx = idxx(:);

subj_ids = unique(idxx);

box_data = [];
group = [];
labels = {};

cnt = 0;

for s = 1:length(subj_ids)

    rows = idxx == subj_ids(s);

    % Hold
    cnt = cnt + 1;
    box_data = [box_data; plv_hold_all(rows)];
    group = [group; cnt * ones(sum(rows),1)];
    labels{cnt} = sprintf('S%d Hold', subj_ids(s));

    % Move
    cnt = cnt + 1;
    box_data = [box_data; plv_move_all(rows)];
    group = [group; cnt * ones(sum(rows),1)];
    labels{cnt} = sprintf('S%d Move', subj_ids(s));

end

figure; hold on;

boxplot(box_data, group, ...
    'Labels', labels, ...
    'Symbol', '', ...
    'Colors', 'k');   % outlines initially black

ylabel('PAC / PLV value')
set(gca, 'FontSize', 14)
xtickangle(45)
box off

% Colors
hold_color = [0 0.4470 0.7410];        % blue
move_color = [0.8500 0.3250 0.0980];   % orange

% Get box handles
h_box = findobj(gca, 'Tag', 'Box');
h_box = flipud(h_box);   % correct order: S1 Hold, S1 Move, S2 Hold, ...

% Add colored patches behind boxes
patch_handles = gobjects(2,1);

for k = 1:length(h_box)

    if mod(k,2) == 1
        this_color = hold_color;
    else
        this_color = move_color;
    end

    p = patch(get(h_box(k),'XData'), get(h_box(k),'YData'), this_color, ...
        'FaceAlpha', 0.35, ...
        'EdgeColor', this_color, ...
        'LineWidth', 1.5);

    % save handles for legend
    if k == 1
        patch_handles(1) = p;
    elseif k == 2
        patch_handles(2) = p;
    end
end

% Make median lines black and thicker
h_med = findobj(gca, 'Tag', 'Median');
set(h_med, 'Color', 'k', 'LineWidth', 2);

% Make whiskers/caps black too
set(findobj(gca, 'Tag', 'Whisker'), 'Color', 'k', 'LineWidth', 1.2);
set(findobj(gca, 'Tag', 'Upper Adjacent Value'), 'Color', 'k', 'LineWidth', 1.2);
set(findobj(gca, 'Tag', 'Lower Adjacent Value'), 'Color', 'k', 'LineWidth', 1.2);

% Put colored patches behind black median lines
uistack(h_med, 'top');

% Legend on top distinguishing Hold vs Move
legend(patch_handles, {'Hold', 'Move'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'Box', 'off');
plot_beautify
ylim([-0.05 0.9])
yticks([0:.1:1])
ylabel('LFO-hG PAC')

% stats
plv_hold_all = plv_hold_all(:);
plv_move_all = plv_move_all(:);
idxx = idxx(:);

subj_ids = unique(idxx);

pvals = nan(length(subj_ids),1);
stats_out = cell(length(subj_ids),1);

for s = 1:length(subj_ids)

    rows = idxx == subj_ids(s);

    hold_vals = plv_hold_all(rows);
    move_vals = plv_move_all(rows);

    % remove NaNs if needed
    good = ~isnan(hold_vals) & ~isnan(move_vals);
    hold_vals = hold_vals(good);
    move_vals = move_vals(good);

    [p, h, stats] = signrank(move_vals, hold_vals);

    pvals(s) = p;
    stats_out{s} = stats;

    fprintf('Subject %d: p = %.4g, zval = %.3f, n = %d\n', ...
        subj_ids(s), p, stats.zval, length(move_vals));

end


%% Mu hold - move 

cd('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves')
addpath(genpath(pwd))
ec176 = load('EC176_Mu_Pow.mat');
ec189 = load('EC189_Mu_Power.mat');
ec210 = load('EC210_Mu_Pow.mat');


% plot now percent channels significant
subj_names = {'ec176','ec189','ec210'};
subj.ec176 = ec176;
subj.ec189 = ec189;
subj.ec210 = ec210;
pow_move_all=[];
pow_hold_all=[];
idxx=[];
for i=1:length(subj_names)

    bad_chI = subj.(subj_names{i}).bad_chI;
    pow_hold = subj.(subj_names{i}).pow_s2;
    pow_move = subj.(subj_names{i}).pow_s3;
    good_ch = find(bad_chI==1);

    a = mean(pow_hold,2);
    pow_hold_all =[pow_hold_all; a(good_ch)];
    

    b = mean(pow_move,2);
    pow_move_all =[pow_move_all; b(good_ch)];
    idxx = [idxx;i*ones(length(good_ch),1)];
end


% now scatter plv values each subject
for i=1:3
    figure;
    idx=find(idxx==i);
    d=[pow_hold_all(idx) pow_move_all(idx)];
    boxplot(d)   
    xticks(1:2)
    xticklabels({'Move','Hold'})
    plot_beautify
    title(subj_names{i})
    %ylim([0 0.9])
    %yticks([0:.1:.8])
end

%all at once
pow_hold_all = pow_hold_all(:);
pow_move_all = pow_move_all(:);
idxx = idxx(:);

subj_ids = unique(idxx);

box_data = [];
group = [];
labels = {};

cnt = 0;

for s = 1:length(subj_ids)

    rows = idxx == subj_ids(s);

    % Hold
    cnt = cnt + 1;
    box_data = [box_data; pow_hold_all(rows)];
    group = [group; cnt * ones(sum(rows),1)];
    labels{cnt} = sprintf('S%d Hold', subj_ids(s));

    % Move
    cnt = cnt + 1;
    box_data = [box_data; pow_move_all(rows)];
    group = [group; cnt * ones(sum(rows),1)];
    labels{cnt} = sprintf('S%d Move', subj_ids(s));

end

figure; hold on;

boxplot(box_data, group, ...
    'Labels', labels, ...
    'Symbol', '', ...
    'Colors', 'k');   % outlines initially black

ylabel('Mu Power')
set(gca, 'FontSize', 14)
xtickangle(45)
box off

% Colors
hold_color = [0 0.4470 0.7410];        % blue
move_color = [0.8500 0.3250 0.0980];   % orange

% Get box handles
h_box = findobj(gca, 'Tag', 'Box');
h_box = flipud(h_box);   % correct order: S1 Hold, S1 Move, S2 Hold, ...

% Add colored patches behind boxes
patch_handles = gobjects(2,1);

for k = 1:length(h_box)

    if mod(k,2) == 1
        this_color = hold_color;
    else
        this_color = move_color;
    end

    p = patch(get(h_box(k),'XData'), get(h_box(k),'YData'), this_color, ...
        'FaceAlpha', 0.35, ...
        'EdgeColor', this_color, ...
        'LineWidth', 1.5);

    % save handles for legend
    if k == 1
        patch_handles(1) = p;
    elseif k == 2
        patch_handles(2) = p;
    end
end

% Make median lines black and thicker
h_med = findobj(gca, 'Tag', 'Median');
set(h_med, 'Color', 'k', 'LineWidth', 2);

% Make whiskers/caps black too
set(findobj(gca, 'Tag', 'Whisker'), 'Color', 'k', 'LineWidth', 1.2);
set(findobj(gca, 'Tag', 'Upper Adjacent Value'), 'Color', 'k', 'LineWidth', 1.2);
set(findobj(gca, 'Tag', 'Lower Adjacent Value'), 'Color', 'k', 'LineWidth', 1.2);

% Put colored patches behind black median lines
uistack(h_med, 'top');

% Legend on top distinguishing Hold vs Move
legend(patch_handles, {'Hold', 'Move'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'Box', 'off');
plot_beautify
ylim([-2 3])
yticks([-2:1:3])
ylabel('Mu Power (z)')
hline(0,'--k')
xticks ''
xticklabels ''

% stats
pow_hold_all = pow_hold_all(:);
pow_move_all = pow_move_all(:);
idxx = idxx(:);

subj_ids = unique(idxx);

pvals = nan(length(subj_ids),1);
stats_out = cell(length(subj_ids),1);

for s = 1:length(subj_ids)

    rows = idxx == subj_ids(s);

    hold_vals = pow_hold_all(rows);
    move_vals = pow_move_all(rows);

    % remove NaNs if needed
    good = ~isnan(hold_vals) & ~isnan(move_vals);
    hold_vals = hold_vals(good);
    move_vals = move_vals(good);

    [p, h, stats] = signrank(move_vals, hold_vals);

    pvals(s) = p;
    stats_out{s} = stats;

    fprintf('Subject %d: p = %.4g, zval = %.3f, n = %d\n', ...
        subj_ids(s), p, stats.zval, length(move_vals));

end


pow_diff = (pow_hold_all - pow_move_all);
m = mean(pow_diff);
mb = sort(bootstrp(1000,@mean,pow_diff));
[mb(25) m mb(975)]

%% PAC of mu and hG 