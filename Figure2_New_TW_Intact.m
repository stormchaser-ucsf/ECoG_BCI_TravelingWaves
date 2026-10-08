
%%% TRAVELING WAVE PAPER FOR NATURAL MOTOR CONTROL (MAIN)
%% init

clc;clear

if ispc
    addpath('C:\Users\nikic\Documents\MATLAB')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\wave-matlab-master\wave-matlab-master'))
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves')

else

    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))

end

disp('init done')

%% PLOT WAVE STATISTICS ACROSS ALL 3 PARTICIPANTS

clc;clear
close all


filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC176_ProcessingForNikhilesh/ecog_data_NN';
filename = 'EC176_hold_waves_stats.mat';
ec176 = load(fullfile(filepath,filename));

filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189';
filename = 'EC189_hold_waves_stats.mat';
ec189 = load(fullfile(filepath,filename));

filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210';
filename = 'EC210_hold_waves_stats.mat';
ec210 = load(fullfile(filepath,filename));

% histogram and boxplot of wave frequency
tmp{1} = ec176.EC176_hold_waves_stats.freq;
tmp{2} = ec189.EC189_hold_waves_stats.freq;
tmp{3} = ec210.EC210_hold_waves_stats.freq;
figure;
hold on
col={'r','b','k'};
xmin = min(cell2mat(tmp));
xmax = max(cell2mat(tmp));
data=NaN(30,3);;
for i=1:3
    x = tmp{i};   
    xi = linspace(xmin,xmax,200);
    [f,xi] = ksdensity(x,xi);
    plot(xi,f,'LineWidth',1,'Color',col{i})
    data(1:length(x),i) = x;    
end
figure;
boxplot(data)
ylim([1 4])
xticks(1:3)
xticklabels({'EC176','EC189','EC210'})
ylabel('No. of Waves/Sec (Freq)')
plot_beautify

%%%% better scatter plot plotting, wave frequency
b1_acc = tmp{1};
b3_acc = tmp{2};
b6_acc = tmp{3};
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;%m11(m11>.25) = NaN;
m22 = b3_acc;%m22(m22>.25) = NaN;
m33 = b6_acc;%m33(m33>.25) = NaN;
y=[nanmean(m11) nanmean(m22) nanmean(m33)];
% scatter ec participants individually
figure; hold on
h=hline(nanmedian(res),'k');
h.LineWidth=3;
h.XData = [0.75 1.25];
bb = sort(bootstrp(1000,@median,res));
[bb(25) median(res) bb(975)]

x=(1:1) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end
xlim([.6 1.4])
ylim([1 4])
xticks ''
plot_beautify
ylabel('Num. of Waves/Sec (Freq)')


%%%% scatter plot of wave duration
tmp{1} = ec176.EC176_hold_waves_stats.dur;
tmp{2} = ec189.EC189_hold_waves_stats.dur;
tmp{3} = ec210.EC210_hold_waves_stats.dur;
b1_acc = tmp{1}*1e3;
b3_acc = tmp{2}*1e3;
b6_acc = tmp{3}*1e3;
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;%m11(m11>.25) = NaN;
m22 = b3_acc;%m22(m22>.25) = NaN;
m33 = b6_acc;%m33(m33>.25) = NaN;
y=[nanmean(m11) nanmean(m22) nanmean(m33)];
% scatter ec participants individually
figure; hold on
h=hline(nanmedian(res),'k');
h.LineWidth=3;
h.XData = [0.75 1.25];
bb = sort(bootstrp(1000,@median,res));
[bb(25) median(res) bb(975)]

x=(1:1) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end
xlim([.6 1.4])
ylim([60 300])
xticks ''
plot_beautify
ylabel('Wave Duration (ms)')


%%%% scatter plot of duty cycle
tmp{1} = ec176.EC176_hold_waves_stats.dcyc;
tmp{2} = ec189.EC189_hold_waves_stats.dcyc;
tmp{3} = ec210.EC210_hold_waves_stats.dcyc;
b1_acc = tmp{1};
b3_acc = tmp{2};
b6_acc = tmp{3};
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;%m11(m11>.25) = NaN;
m22 = b3_acc;%m22(m22>.25) = NaN;
m33 = b6_acc;%m33(m33>.25) = NaN;
y=[nanmean(m11) nanmean(m22) nanmean(m33)];
% scatter ec participants individually
figure; hold on
h=hline(nanmedian(res),'k');
h.LineWidth=3;
h.XData = [0.75 1.25];
bb = sort(bootstrp(1000,@median,res));
[bb(25) median(res) bb(975)]

x=(1:1) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end
xlim([.6 1.4])
ylim([0.1 0.6])
xticks ''
plot_beautify
ylabel('Wave Duty Cycle')


%% COMPARISON BETWEEN WAVE AND NON WAVE EPOCHS
% plv and mu power

clc;clear
close all

filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC176_ProcessingForNikhilesh/ecog_data_NN';
filename = 'EC176_PAC_MuPower_Waves_HoldPeriod.mat';
ec176 = load(fullfile(filepath,filename));

filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189';
filename = 'EC189_PAC_MuPower_Waves_HoldPeriod.mat';
ec189 = load(fullfile(filepath,filename));

filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210';
filename = 'EC210_PAC_MuPower_Waves_HoldPeriod.mat';
ec210 = load(fullfile(filepath,filename));


%%%% scatter plot
% nonwaves
b1_acc = ec176.res_EC176_Mu_hG_plv_waves(:,2)';
b3_acc = ec189.res_EC189_Mu_hG_plv_waves(:,2)';
b6_acc = ec210.res_EC210_Mu_hG_plv_waves(:,2)';
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;
m22 = b3_acc;
m33 = b6_acc;
x=1:3;
y=[mean(m11) mean(m22) mean(m33)];
% scatter B1 and B3 and B6 individually
figure; hold on
h=hline(median(res),'k');
h.LineWidth=3;
h.XData = [0.75 1.25];

x=(1:1) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end


x=(1:1) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end

%%%% waves
b1_acc = ec176.res_EC176_Mu_hG_plv_waves(:,1)';
b3_acc = ec189.res_EC189_Mu_hG_plv_waves(:,1)';
b6_acc = ec210.res_EC210_Mu_hG_plv_waves(:,1)';
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;
m22 = b3_acc;
m33 = b6_acc;
x=1:3;
y=[mean(m11) mean(m22) mean(m33)];
% scatter B1 and B3 and B6 individually
hold on
h=hline(median(res),'k');
h.LineWidth=3;
h.XData = [1.75 2.25];

x=(2) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(2) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end


x=(2) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end

ylim([-0.3 0.5])
yticks([-1:.1:1])
xlim([.5 2.5])
h=hline(0);
set(h,'LineWidth',1)
xticks ''
plot_beautify
%boxplot([bhat_mu bhat_lfo])
ylim([-.2 .5])
xticks(1:2)
xticklabels({'Mu-hG','LFO-hG'})
ylabel('Slope')


%%%% with lines:
%%% plotting
res_days_B1 = ec176.res_EC176_Mu_hG_plv_waves;
res_days_B3 = ec189.res_EC189_Mu_hG_plv_waves;
res_days_B6 = ec210.res_EC210_Mu_hG_plv_waves;


res_days = [res_days_B1;res_days_B3;res_days_B6];
figure;hold on


%b1
idx = [ones(size(res_days_B1,1),1) 2*ones(size(res_days_B1,1),1)];
idx = idx + 0.075*randn(size(idx));
scatter(idx(:,1),res_days_B1(:,1),50,'b','LineWidth',1)
scatter(idx(:,2),res_days_B1(:,2),50,'b',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B1(i,1),res_days_B1(i,2)],...
        'Color',[0 0 1 .35])
end

%b3
%figure;hold on
idx = [ones(size(res_days_B3,1),1) 2*ones(size(res_days_B3,1),1)];
idx = idx + 0.075*randn(size(idx));
scatter(idx(:,1),res_days_B3(:,1),50,'r','LineWidth',1)
scatter(idx(:,2),res_days_B3(:,2),50,'r',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B3(i,1),res_days_B3(i,2)],...
        'Color',[1 0 0 .35])
end

%b6
%figure;hold on
idx = [ones(size(res_days_B6,1),1) 2*ones(size(res_days_B6,1),1)];
idx = idx + 0.075*randn(size(idx));
scatter(idx(:,1),res_days_B6(:,1),50,'k','LineWidth',1)
scatter(idx(:,2),res_days_B6(:,2),50,'k',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B6(i,1),res_days_B6(i,2)],...
        'Color',[0.25 0.25 0.25 .35])
end

h=hline(nanmean(res_days(:,1)),'k');
h.LineWidth=3;
h.XData = [0.75 1.25];

h=hline(nanmean(res_days(:,2)),'k');
h.LineWidth=3;
h.XData = [1.75 2.25];

ylim([0.33 0.46])
xlim([.5 2.5])
xticks(1:2)
xticklabels({'Inaccurate Trials','Accurate Trials'})
ylabel('Wave duty cycle')
plot_beautify

% overall stats
[p,h] = signrank(res_days(:,1),res_days(:,2))

%%%% all in one boxplots
res_days_B1 = ec176.res_EC176_Mu_hG_plv_waves;
res_days_B3 = ec189.res_EC189_Mu_hG_plv_waves;
res_days_B6 = ec210.res_EC210_Mu_hG_plv_waves;
res = [res_days_B1;res_days_B3;res_days_B6];
idx=[ones(length(res_days_B1),1);2*ones(length(res_days_B3),1);...
    3*ones(length(res_days_B6),1)];


[p,h,stats] = signrank(res_days_B1(:,1),res_days_B1(:,2))
[p,h,stats] = signrank(res_days_B3(:,1),res_days_B3(:,2))
[p,h,stats] = signrank(res_days_B6(:,1),res_days_B6(:,2))

figure; hold on;
colors = [1 0 0;    % Wave: red
          0 0 1];   % Non-wave: blue
positions = [1 2; 4 5; 7 8];
for s = 1:3 %subject
    for c = 1:2 % condition

        data = res(idx == s, c);
        data = data(~isnan(data));

        % Draw boxplot
        boxplot(data, ...
            'Positions', positions(s,c), ...
            'Widths', 0.65, ...
            'Colors', 'k', ...
            'Symbol', '');

        % Fill box with color
        h = findobj(gca, 'Tag', 'Box');
        patch(get(h(1), 'XData'), ...
              get(h(1), 'YData'), ...
              colors(c,:), ...
              'FaceAlpha', 0.5, ...
              'EdgeColor', 'k');
    end
end

% Subject labels
set(gca, 'XTick', mean(positions,2), ...
         'XTickLabel', {'EC176','EC189','EC210'});
ylabel('Mu - hG PAC');
xlim([0 9]);
box off;
set(gca, 'FontSize', 12);
ylim([.25 .55])

% Legend
h1 = patch(nan,nan,colors(1,:), 'FaceAlpha',0.5);
h2 = patch(nan,nan,colors(2,:), 'FaceAlpha',0.5);
legend([h1 h2], {'Wave','Non-wave'}, 'Location','best');
plot_beautify

%%%% all in one boxplots for mu power
% res_days_B1 = [mean(ec176.res_EC176_mu_pow_waves,1)' mean(ec176.res_EC176_mu_pow_non_waves,1)'];
% res_days_B3 = [mean(ec189.res_EC189_mu_pow_waves,1)' mean(ec189.res_EC189_mu_pow_non_waves,1)'];
% res_days_B6 = [mean(ec210.res_EC210_mu_pow_waves,1)' mean(ec210.res_EC210_mu_pow_non_waves,1)'];
res_days_B1 = [mean(ec176.res_EC176_mu_pow_waves,2) mean(ec176.res_EC176_mu_pow_non_waves,2)];
res_days_B3 = [mean(ec189.res_EC189_mu_pow_waves,2) mean(ec189.res_EC189_mu_pow_non_waves,2)];
res_days_B6 = [mean(ec210.res_EC210_mu_pow_waves,2) mean(ec210.res_EC210_mu_pow_non_waves,2)];
res = [res_days_B1;res_days_B3;res_days_B6];
idx=[ones(length(res_days_B1),1);2*ones(length(res_days_B3),1);...
    3*ones(length(res_days_B6),1)];

[p,h,stats] = signrank(res_days_B1(:,1),res_days_B1(:,2))
[p,h,stats] = signrank(res_days_B3(:,1),res_days_B3(:,2))
[p,h,stats] = signrank(res_days_B6(:,1),res_days_B6(:,2))

figure; hold on;
colors = [1 0 0;    % Wave: red
          0 0 1];   % Non-wave: blue
positions = [1 2; 4 5; 7 8];
for s = 1:3 %subject
    for c = 1:2 % condition

        data = res(idx == s, c);
        data = data(~isnan(data));

        % Draw boxplot
        boxplot(data, ...
            'Positions', positions(s,c), ...
            'Widths', 0.65, ...
            'Colors', 'k', ...
            'Symbol', '');

        % Fill box with color
        h = findobj(gca, 'Tag', 'Box');
        patch(get(h(1), 'XData'), ...
              get(h(1), 'YData'), ...
              colors(c,:), ...
              'FaceAlpha', 0.5, ...
              'EdgeColor', 'k');
    end
end

% Subject labels
set(gca, 'XTick', mean(positions,2), ...
         'XTickLabel', {'EC176','EC189','EC210'});
ylabel('Mu pow (z)');
xlim([0 9]);
box off;
set(gca, 'FontSize', 12);
ylim([.2 1])
plot_beautify

% Legend
h1 = patch(nan,nan,colors(1,:), 'FaceAlpha',0.5);
h2 = patch(nan,nan,colors(2,:), 'FaceAlpha',0.5);
legend([h1 h2], {'Wave','Non-wave'}, 'Location','best');
plot_beautify

%% plot example traveling waves for EC189 in hold period


%%%%%% loading data
get_data_analyses_EC189
imaging_EC189
close all

%%%%%%% filters and such
cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189/')
ecog_grid=[];
k=1;
for i=1:16:256
    ecog_grid(k,:) = i:i+15;
    k=k+1;
end
ecog_grid = flipud(ecog_grid);

load('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/20251203/Robot3DArrow/133334/Imagined/Data0001.mat')
Params=TrialData.Params;
filterbank=[];k=1;
for i=9:16
    [b,a]=butter(3,Params.FilterBank(i).fpass/(Fs/2));
    filterbank(k).b=b;
    filterbank(k).a=a;
    filterbank(k).fpass=Params.FilterBank(i).fpass;
    k=k+1;
end

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',5.5,'HalfPowerFrequency2',8.5, ...
    'SampleRate',Fs);
hGFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',Fs);

%%% parallel check
parallel_check

%%%% MU TRAVELING WAVE ANALYSES
ch=1:256;

hg = filtfilt(hGFilt,zscore(lfp(:,ch)));
hg = abs(hilbert(hg));
hg_mu = filtfilt(bpFilt,hg);
hg_mu  = hilbert(hg_mu);

mu_signal = filtfilt(bpFilt,zscore(lfp(:,ch)));
mu_wave = hilbert(mu_signal);
mu_pow = abs(mu_wave);
%mu_wave(:,bad_ch) = 1e-8*randn(size(mu_wave(:,bad_ch)));


% mask
mask = bad_chI(ecog_grid);

%analyze
dcyc_move = [];
dcyc_hold = [];
wave_f = [];
wave_dur = [];
mu_pow_waves=[];
mu_pow_non_waves=[];
stats={};
for i=1:length(trial_timings)
    timings = [trial_timings(i).movement.cue.time];
    % for j=1:length(timings)
    %     vline(timings(1),'r') % rest
    %     vline(timings(2),'y') % hold
    %     vline(timings(3),'g') % go
    % end


    %%%% HOLD PERIOD %%%%%%
    %%%% main
    st = trial_timings(i).movement.cue(2).time; %anin    
    %st = st+0.3;
    %stp=st+2.5;
    stp = trial_timings(i).movement.cue(3).time; %anin    
    %%%% main end

    %%% for movement
    % st = trial_timings(i).movement.cue(3).time; %anin  
    % st=st+0.5;
    % stp = st+3;

    % if stp-st > 3
    %     stp=st+3;
    % end
    index = (lfp_time >= st) .* (lfp_time<=stp);
    data = mu_wave(logical(index),:); % hilbert of mu
    data_pow = mu_pow(logical(index),:); %mu amplitude
    hg_mu_data = hg_mu(logical(index),:); %hilbert of hg mu

    % downsample to 50Hz
    tx = (1/Fs)*[0:size(data,1)-1];
    df = resample(data,tx,50); %hilbert of mu
    df_pow = resample(data_pow,tx,50); %mu amplitude
    hg_mu_data = resample(hg_mu_data,tx,50); %hilbert of hg mu
    
    % detect waves based on spatiotemporally stable phase gradients    
    planar_val_time=[];
    parfor t=1:size(df,1)        
        % estimate planar waves across mini grid
        tmp = df(t,:);
        tmp(bad_ch)= NaN + 1i*NaN;
        xph = tmp(ecog_grid);
        %[planar_val,aa,bb] = planar_stats_muller_intact(xph,mask);        
        [planar_val,aa,bb] = planar_stats_muller(xph);   
        planar_val(mask==0) = NaN +1i*NaN;
        planar_val_time(t,:,:) = planar_val;
    end

    stab=[];
    for k=2:size(planar_val_time,1)
        xt = planar_val_time(k,:,:);xt=xt(:);
        xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
        stab(k-1) = - nanmean(abs(xt - xtm1));
    end
    stab1 = zscore(stab(1:end));
    [out,st,stp] = wave_stability_detect(stab1,0,3);

    % figure;plot(stab1)
    % hline(0)

    t = length(stab1) * 20/1e3;
    ff1 = length(out)/t; % frequency/s
    d = mean(out) * 20/1e3; %duration in s
    dcyc=ff1*d;
    dcyc_hold =  [dcyc_hold dcyc];
    wave_f(i) = ff1;
    wave_dur(i) = d;


    %%%% get mu power and plv within wave periods
    tmp={};tmp_mu={};I=ones(length(stab1)+1,1);
    tmp_hg_mu={};
    for k=1:length(st)
        tmp{k} = df_pow(st(k):stp(k),bad_chI); %mu amplitude
        tmp_mu{k} = df(st(k):stp(k),bad_chI); %hilbert of mu
        tmp_hg_mu{k} = hg_mu_data(st(k):stp(k),bad_chI); %hilbert of hg mu
        I(st(k):stp(k))=0;
    end
    % power
    tmp = cell2mat(tmp');
    mu_pow_waves = [mu_pow_waves ;nanmean(tmp,1)];

    %plv
    tmp_mu = angle(cell2mat(tmp_mu'));
    tmp_hg_mu = angle(cell2mat(tmp_hg_mu'));
    res_wave = (exp(1i .* (tmp_mu - tmp_hg_mu)));
    stats(i).plv_wave = res_wave;


    %%%% get mu power within non wave periods
    I = logical(I);
    % power
    tmp_nonwave = df_pow(I,bad_chI);
    mu_pow_non_waves = [mu_pow_non_waves ;nanmean(tmp_nonwave,1)];
    % plv
    tmp_mu = angle(df(I,bad_chI));
    tmp_hg_mu = angle(hg_mu_data(I,bad_chI));
    res_nonwave = (exp(1i .* (tmp_mu - tmp_hg_mu)));
    stats(i).plv_nonwave = res_nonwave;




    %%%%% MOVE PERIOD %%%%%%
    st = trial_timings(i).movement.cue(3).time; %anin
    st=st+0.5;
    stp = st+3;    
    index = (lfp_time >= st) .* (lfp_time<=stp);
    data = mu_wave(logical(index),:);

    % downsample to 50Hz
    tx = (1/Fs)*[0:size(data,1)-1];
    df = resample(data,tx,50);

    % detect waves based on spatiotemporally stable phase gradients    
    planar_val_time=[];
    parfor t=1:size(df,1)        
        % estimate planar waves across mini grid
        tmp = df(t,:);
        tmp(bad_ch)= NaN + 1i*NaN;
        xph = tmp(ecog_grid);
        %[planar_val,aa,bb] = planar_stats_muller_intact(xph,mask);        
        [planar_val,aa,bb] = planar_stats_muller(xph);        
        planar_val(mask==0) = NaN +1i*NaN;
        planar_val_time(t,:,:) = planar_val;
    end

    stab=[];
    for k=2:size(planar_val_time,1)
        xt = planar_val_time(k,:,:);xt=xt(:);
        xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
        stab(k-1) = - nanmean(abs(xt - xtm1));
    end
    stab1 = zscore(stab(1:end));
    [out,st,stp] = wave_stability_detect(stab1,0,3);
    t = length(stab1) * 20/1e3;
    ff1 = length(out)/t; % frequency/s
    d = mean(out) * 20/1e3; %duration in s
    dcyc=ff1*d;
    dcyc_move =  [dcyc_move dcyc];
end

figure;
boxplot([ dcyc_hold(:) dcyc_move(:)])
[p,h] = signrank(dcyc_hold,dcyc_move)

figure;
boxplot([mean(mu_pow_waves,2) mean(mu_pow_non_waves,2)])
p=signrank(mean(mu_pow_waves,2)', mean(mu_pow_non_waves,2)');
xticks(1:2)
xticklabels({'Wave','Non wave'})
ylabel('Mu power')
title(['pval of ' num2str(p)])
plot_beautify

res = get_plv_stats_intact(stats);
figure;boxplot(res(:,1)-res(:,2));hline(0)


EC189_hold_waves_stats.freq = wave_f;
EC189_hold_waves_stats.dur = wave_dur;
EC189_hold_waves_stats.dcyc = dcyc_hold;
save EC189_hold_waves_stats EC189_hold_waves_stats -v7.3

res_EC189_Mu_hG_plv_waves = res;
res_EC189_mu_pow_waves = mu_pow_waves;
res_EC189_mu_pow_non_waves = mu_pow_non_waves;
save EC189_PAC_MuPower_Waves_HoldPeriod res_EC189_Mu_hG_plv_waves ...
    res_EC189_mu_pow_waves res_EC189_mu_pow_non_waves -v7.3


%%%%% PLOT SOME EXAMPLE TRAVELING WAVES SPATIAL GRADIENTS
% plotting of phase gradients
idx=59:61;
%idx = 9:11;
%idx = 42:44;
%idx = 69:71;
for i=1:length(idx)
    planar_val = squeeze(planar_val_time(idx(i),:,:));
    M = real(planar_val);
    N = imag(planar_val);
    [XX,YY] = meshgrid( 1:size(planar_val,2), 1:size(planar_val,1) );
    figure;
    quiver(XX,YY,M,N);axis tight
    set(gca,'Ydir','reverse')
    axis off
    plot_beautify
end

%%%% (MAIN) %%%%
% stats on the curl of a traveling wave -> roughly portion center portion
% of entire grid
%cols=[3:9];
%rows=[5:11];
cols = 13+[-3:3];
rows = 4+[-3:3];
wave_data = df(53:78,:);
% compute curl at each time point
curl_coeff=[];
pvals=[];
pvals_perm=[];
yc = 4;
xc = 4;
[X,Y] = meshgrid(1:length(cols),1:length(rows));
d = max(abs(X-xc),abs(Y-yc)); % Chebyshev distance from center -> square rings
rings = unique(d(:));
for i=1:size(wave_data,1)

    tmp = wave_data(i,:);
    xph = tmp(ecog_grid);
    xph = xph(rows,cols);
    [planar_val,aa,bb,xphs] = planar_stats_muller(xph);
    M = real(planar_val);
    N = imag(planar_val);
    [XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );
    [curl_val0] = curl(XX,YY,M,N);

    pl = angle(xph);
    %pl = angle(xphs);
    cl = curl_val0;

    [cc,pv,center_point] = ...
        phase_correlation_rotation( pl, cl,[xc,yc],1 ); %pl -> phase map, cl -> curl map
    curl_coeff(i) = cc;
    pvals(i) = pv;

    % to get pvalue from null permutation testing
    boot_val=[];
    parfor iter=1:1000
        xph_shuff = xph;
        for r = 1:length(rings)
            idx = find(d == rings(r));
            % shuffle phase values within this square ring
            xph_shuff(idx) = xph(idx(randperm(length(idx))));
        end
        pl = angle(xph_shuff);
        [cc,pv,center_point] = ...
            phase_correlation_rotation( pl, cl,[xc,yc],1 ); %pl -> phase map, cl -> curl map
        boot_val(iter) = cc;
    end
    pvals_perm(i) = max(1/length(boot_val),...
        sum(boot_val>abs(curl_coeff(i)))/length(boot_val));
end
curl_coeff=abs(curl_coeff);
median(curl_coeff)
figure;
boxplot(curl_coeff)
ylim([0.4 1])
ylabel('Circular-circular R')
plot_beautify
xlim([.85 1.15])
xticks ''

[pfdr,pp] = fdr(pvals_perm,0.05);pfdr
sum(pvals_perm<=pfdr)/length(pvals_perm)

%%%%% MAIN PLOTTING CODE %%%%
xx = df(53:78,:); %103 to 118 get the start and stop from out,st,stp
xx = resample(xx,4,1);
tt = [0:(size(xx,1)-1)]*(1000/(50*4));
%clim = [min(xx(:)) max(xx(:))];
figure;
colormap hot


clf
v = VideoWriter('EC189_WaveExample.avi','Motion JPEG AVI');
v.FrameRate = 20;
v.Quality=100;
open(v);
data_wav=[];
for t = 1:size(xx,1)

    tmp = (xx(t,:));
    xph = tmp(ecog_grid);
    M = real(xph);
    N = imag(xph);
    [tmp,s1] = smoothn({M,N},'robust');
    M = tmp{1}; N = tmp{2};
    xph = M +1j*N;
    xph = cos(angle(xph));
    %xph = M;
    xph = xph(rows,cols);
    imagesc(xph)
    axis off
    %axis image;
    %colorbar;
    %caxis(clim);
    caxis([-1 1])
    shading interp
    data_wav = cat(3,data_wav,xph);

    title(sprintf('Time in ms %d',tt(t)));
    drawnow;
    frame = getframe(gcf);
    writeVideo(v,frame);
end

close(v);

cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/mat_plots/WaveExamples/EC189')
% now make a bunch of plots
% for t = 1:size(xx,1)
% 
%     tmp = (xx(t,:));
%     xph = tmp(ecog_grid);
%     M = real(xph);
%     N = imag(xph);
%     [tmp,s1] = smoothn({M,N},'robust');
%     M = tmp{1}; N = tmp{2};
%     xph = M +1j*N;
%     xph = cos(angle(xph));
%     %xph = M;
%     xph = xph(rows,cols);
%     h=figure;
%     colormap hot
%     imagesc(xph)
%     axis off
%     %axis image;
%     %colorbar;
%     %caxis(clim);
%     caxis([-1 1])
%     shading interp
%     %data_wav = cat(3,data_wav,xph);
% 
%     title(sprintf('Time in ms %d',tt(t)));
% 
%     % save MATLAB figure
%     tmp_title = ['Time ' num2str(tt(t)) 's_v2.fig'];
%     tmp_title1 = ['Time ' num2str(tt(t)) 's_v2.png'];
%     savefig(h,tmp_title);
% 
%     % save PNG
%     exportgraphics(h,tmp_title1,'Resolution',300);
% end

%%%%% with quivers (MAIN) %%%%%%
% upsample the data in specific periods
xx = df(53:78,:); %64to 74
xx = resample(xx,4,1);

tt = (0:(size(xx,1)-1))*(1000/(50*4));

%cols=[1:4];
%rows=[1:11];
%cols=[11:18];
%rows = [1:11];

% ===================== VIDEO =====================

hfig = figure;
%set(gca,'Ydir','reverse')
colormap hot

v = VideoWriter('EC189_WaveExample_quivers.avi','Motion JPEG AVI');
v.FrameRate = 10;
v.Quality = 100;
open(v);

data_wav = [];

for t = 1:size(xx,1)

    tmp = xx(t,:);

    % Arrange channels spatially
    xph = tmp(ecog_grid);

    % Real/imaginary components
    M = real(xph);
    N = imag(xph);

    % Spatial smoothing
    [tmp_sm,s1] = smoothn({M,N},'robust');
    M = tmp_sm{1};
    N = tmp_sm{2};

    % Reconstruct smoothed complex field
    xph_complex = M + 1j*N;

    % Phase
    ph = angle(xph_complex);

    % Imagesc quantity
    xph_img = cos(ph);


    [pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN(xph_complex, ...
        1,-1);
    ph=pd;
    M =  pm.*cos(ph);
    N =  pm.*sin(ph);
    [tmp,s2] = smoothn({M,N},'robust');
    M = tmp{1}; N = tmp{2};
    planar_val = M + 1j*N;
    planar_val = planar_val*-1;

    % phase gradient
    M = real(planar_val);
    N = imag(planar_val);
    U = M;
    V = N;

    %[XX,YY] = meshgrid( 1:size(planar_val,2), 1:size(planar_val,1) );
    % figure;
    % quiver(XX,YY,M,N);axis tight

    % Quiver components -- unit vectors representing phase
    % U = cos(ph);
    % V = sin(ph);

    % Crop spatial region
    xph_img = xph_img(rows,cols);
    U = U(rows,cols);
    V = V(rows,cols);

    % Coordinates
    [XX,YY] = meshgrid(1:size(xph_img,2), ...
        1:size(xph_img,1));

    clf(hfig)

    % Background phase image
    imagesc(xph_img);
    caxis([-1 1]);
    axis off
    axis image
    colormap hot

    hold on


    % Quiver overlay
    quiver(XX,YY,U,V, ...
        'm', ...
        'LineWidth',1.5, ...
        'AutoScale','on', ...
        'AutoScaleFactor',0.7);

    hold off

    data_wav = cat(3,data_wav,xph_img);

    title(sprintf('Time in ms %.0f',tt(t)));

    drawnow;

    frame = getframe(hfig);
    writeVideo(v,frame);
end

close(v);


% ===================== INDIVIDUAL FIGURES =====================

cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/mat_plots/WaveExamples/EC189/plots')
for t = 1:size(xx,1)

    tmp = xx(t,:);

    % Arrange channels spatially
    xph = tmp(ecog_grid);

    M = real(xph);
    N = imag(xph);

    % Smooth real and imaginary components
    [tmp_sm,s1] = smoothn({M,N},'robust');

    M = tmp_sm{1};
    N = tmp_sm{2};

    % Complex field
    xph_complex = M + 1j*N;

    % Phase
    ph = angle(xph_complex);

    % Image
    xph_img = cos(ph);

    % phase gradient computations
    [pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN(xph_complex, ...
        1,-1);
    ph=pd;
    M =  pm.*cos(ph);
    N =  pm.*sin(ph);
    [tmp,s2] = smoothn({M,N},'robust');
    M = tmp{1}; N = tmp{2};
    planar_val = M + 1j*N;
    planar_val = planar_val*-1;

    % phase gradient
    M = real(planar_val);
    N = imag(planar_val);
    U = M;
    V = N;

    % Crop
    xph_img = xph_img(rows,cols);
    U = U(rows,cols);
    V = V(rows,cols);

    % Grid coordinates
    [XX,YY] = meshgrid(1:size(xph_img,2), ...
        1:size(xph_img,1));

    % Figure
    h = figure;
    colormap hot

    imagesc(xph_img);
    caxis([-1 1]);

    axis image
    axis off

    hold on

    quiver(XX,YY,U,V, ...
        'm', ...
        'LineWidth',1.5, ...
        'AutoScale','on', ...
        'AutoScaleFactor',0.7);

    hold off

    title(sprintf('Time in ms %.0f',tt(t)));

    % Save MATLAB figure
    tmp_title  = ['Time_' num2str(tt(t),'%.0f') 'ms_v2.fig'];
    tmp_title1 = ['Time_' num2str(tt(t),'%.0f') 'ms_v2.png'];

    savefig(h,tmp_title);

    % Save PNG
    exportgraphics(h,tmp_title1,'Resolution',300);

    close(h);
end
%%%%%



%% PLOTTING AS SVG


cd('C:\Users\nikic\Documents\Ganguly lab\ECoG BCI\BCI_Paper_Waves_Hand\Paper_text\Figures_New\Figure6\')

% svg
set(gcf,'PaperPositionMode','auto');
print(gcf,'NaturalMotorControl_Waves_NonWaves_Mu_hG_PAC.svg','-dsvg','-painters','-r300');
    
% png
set(gcf,'PaperPositionMode','auto');
print(gcf,'B3_BrainElec.png','-dpng','-r500');





