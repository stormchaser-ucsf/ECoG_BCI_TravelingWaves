
%%%% PLOT DECODING ACCURACY IN HG FOR WAVES VS. NON WAVES EPOCHS
% PLOT DUTY CYCLE IN CORRECT VS. INCORRECT TRIALS

%% init


clear
clc


if ispc
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves'))
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
else

    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))
    cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves')
end
disp('init done')


%% HG DECODING ACCURACIES DURING WAVES VS. NON WAVE EPOCHS

%b1
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
cd(root_path)
load hg_wave_nonwave_MLP_3DArrow_CL_v2


res=[];res_bin=[];
for i=1:size(acc_nonwave,1)
    %tmp0=squeeze((acc_nonwave_least(i,:,:)));
    tmp=squeeze((acc_nonwave(i,:,:)));
    tmp1=squeeze((acc_wave(i,:,:)));
    %res(i,:) = [mean(diag(tmp0)) mean(diag(tmp)) mean(diag(tmp1))];
    res(i,:) = [ mean(diag(tmp)) mean(diag(tmp1))];

    tmp=squeeze((acc_bin_nonwave(i,:,:)));
    tmp1=squeeze((acc_bin_wave(i,:,:)));
    res_bin(i,:) = [mean(diag(tmp)) mean(diag(tmp1))];
end
figure;boxplot(100*res)
xticks(1:2)
xticklabels({'Non wave epochs', 'Wave epochs'})
%xticklabels({'Most unstable nowave','Non wave epochs', 'Wave epochs'})
title('B1')
[p,h,stats]=signrank(res(:,2),res(:,1),'method','approximate');
[p stats.zval]
plot_beautify
res_B1=res;
ylabel('Trial level Decoding Accuracy')
res_bin_B1= res_bin;

%b3
clearvars -except res_B3 res_B1 res_bin_B1
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
cd(root_path)
load hg_wave_nonwave_MLP_3DArrow_CL_v2

res=[];res_bin=[];
for i=1:size(acc_nonwave,1)
    %tmp0=squeeze((acc_nonwave_least(i,:,:)));
    tmp=squeeze((acc_nonwave(i,:,:)));
    tmp1=squeeze((acc_wave(i,:,:)));
    %res(i,:) = [mean(diag(tmp0)) mean(diag(tmp)) mean(diag(tmp1))];
    res(i,:) = [ mean(diag(tmp)) mean(diag(tmp1))];

    tmp=squeeze((acc_bin_nonwave(i,:,:)));
    tmp1=squeeze((acc_bin_wave(i,:,:)));
    res_bin(i,:) = [mean(diag(tmp)) mean(diag(tmp1))];
end
figure;boxplot(100*res,'Whisker',2)
xticks(1:2)
xticklabels({'Non wave epochs', 'Wave epochs'})
%xticklabels({'Most unstable nowave','Non wave epochs', 'Wave epochs'})
title('B3')
[p,h,stats]=signrank(res(:,2),res(:,1),'method','approximate');
[p stats.zval]
plot_beautify
res_B3=res;
ylabel('Trial level Decoding Accuracy')
res_bin_B3= res_bin;

%b6
clearvars -except res_B3 res_B1 res_bin_B3 res_bin_B1
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
cd(root_path)
%%load hg_wave_nonwave_MLP_3DArrow_CL_AllData_v3_AllFolders
%load hg_wave_nonwave_MLP_3DArrow_CL_AllData_v4_AllFolders
load hg_wave_nonwave_MLP_3DArrow_CL_AllData_v5_AllFolders

res=[];res_bin=[];
for i=1:size(acc_nonwave,1)
    %tmp0=squeeze((acc_nonwave_least(i,:,:)));
    tmp=squeeze((acc_nonwave(i,:,:)));
    tmp1=squeeze((acc_wave(i,:,:)));
    %res(i,:) = [mean(diag(tmp0)) mean(diag(tmp)) mean(diag(tmp1))];
    res(i,:) = [ mean(diag(tmp)) mean(diag(tmp1))];

    tmp=squeeze((acc_bin_nonwave(i,:,:)));
    tmp1=squeeze((acc_bin_wave(i,:,:)));
    res_bin(i,:) = [mean(diag(tmp)) mean(diag(tmp1))];
end
figure;boxplot(100*res,'Whisker',2)
xticks(1:2)
xticklabels({'Non wave epochs', 'Wave epochs'})
%xticklabels({'Most unstable nowave','Non wave epochs', 'Wave epochs'})
title('B6')
[p,h,stats]=signrank(res(:,2),res(:,1),'method','approximate');
[p stats.zval]
plot_beautify
res_B6=res;
ylabel('Trial level Decoding Accuracy')
res_bin_B6=res_bin;

% 
% res_bin =[res_bin_B1;res_bin_B3;res_bin_B6];
% %res_bin =[res_bin_B6];
% figure;
% boxplot(res_bin,'notch','on')
% ylim([0.45 0.625])
% signrank(res_bin(:,1),res_bin(:,2))

res_bin =[res_bin_B1;res_bin_B3;res_bin_B6];
%res_bin =[res_bin_B6];
figure;
boxplot(res_bin,'notch','on')
%ylim([0.45 0.625])
signrank(res_bin(:,1),res_bin(:,2))
title('Bin level accuracy')

%% DUTY CYCLE ANALYSES

clc;clear
close all

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
load('B1_waves_stability_hgFilterBank_PLV_AccStatsCL_v2.mat','stats_cl_days')
[res_days_B1, res_days_f_B1, res_days_d_B1] = get_duty_cycle(stats_cl_days);

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
load('B3_waves_3DArrow_stability_hgFilterBank_PLV_AccStatsCL_v2.mat','stats_cl_days')
[res_days_B3, res_days_f_B3, res_days_d_B3] = get_duty_cycle(stats_cl_days);

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
load('B6_waves_stability_hgFilterBank_PLV_AccStatsCL_v2_AllData.mat','stats_cl_days')
[res_days_B6, res_days_f_B6, res_days_d_B6] = get_duty_cycle(stats_cl_days(1:end-1));

% res_days_B1 = log(res_days_B1);
% res_days_B3 = log(res_days_B3);
% res_days_B6 = log(res_days_B6);

%close all

%%% plotting
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

%%%% STATS
% lmm
subj_idx = [ones(size(res_days_B1,1),1);2*ones(size(res_days_B3,1),1);...
    3*ones(size(res_days_B6,1),1)];
%bhat = [bhat_mu bhat_lfo];bhat=bhat(:);
dcyc = res_days(:);
oscillation_type = categorical([zeros(size(res_days,1),1);...
    ones(size(res_days,1),1)]);
subject = categorical([subj_idx;subj_idx]);
data = table(dcyc,oscillation_type,subject);
glm = fitlme(data,'dcyc ~ 1+(oscillation_type) + (1|subject)')

% difference with CI via bootstrap
a = res_days(:,1) - res_days(:,2);
ab = sort(bootstrp(1000,@mean,a));
[ab(25) mean(a) ab(975)]
% pval
stat = glm.Coefficients.tStat(2);
boot=[];
bhat_mu  = res_days(:,1);
bhat_lfo  = res_days(:,2);
parfor i=1:1000

    oscillation_type_rand=[];
    subject_rnd = [];
    pac_rnd=[];
    for ii=1:3
        idx = find(subj_idx==ii);
        mu_tmp = bhat_mu(idx);
        lfo_tmp = bhat_lfo(idx);
        subj_tmp = subj_idx(idx);
        osc_tmp = categorical([zeros(size(mu_tmp));ones(size(lfo_tmp))]);
        osc_tmp = osc_tmp(randperm(numel(osc_tmp)));
        
        subject_rnd = [subject_rnd;[subj_tmp;subj_tmp]];
        pac_rnd = [pac_rnd;[mu_tmp;lfo_tmp]];
        oscillation_type_rand = [oscillation_type_rand;osc_tmp];
    end   
    
    data_rnd = table(pac_rnd,oscillation_type_rand,subject_rnd);
    glm_rnd = fitlme(data_rnd,...
        'pac_rnd ~ 1+(oscillation_type_rand) + (1|subject_rnd)');
    boot(i) = glm_rnd.Coefficients.tStat(2);
end

figure;hist(boot)
vline(stat)
max(1/length(boot),sum(boot>abs(stat))/length(boot))

% WSRT
stats=[];
pval=[];
for ii=1:3
    idx = find(subj_idx==ii);
    a = bhat_mu(idx);
    b = bhat_lfo(idx);
    [p,h,s] = signrank(b,a,'method','approximate');
    stats(ii) = s.zval;
    pval(ii) = p;
end
stats
pval


%% PLOTTING AS SVG


cd('C:\Users\nikic\Documents\Ganguly lab\ECoG BCI\BCI_Paper_Waves_Hand\Paper_text\Figures_New\Figure6\')

% svg
set(gcf,'PaperPositionMode','auto');
print(gcf,'WaveDutyCycle.svg','-dsvg','-painters','-r300');
    
% png
set(gcf,'PaperPositionMode','auto');
print(gcf,'B3_BrainElec.png','-dpng','-r500');





