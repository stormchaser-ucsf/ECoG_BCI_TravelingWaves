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
clc;clear
close all



