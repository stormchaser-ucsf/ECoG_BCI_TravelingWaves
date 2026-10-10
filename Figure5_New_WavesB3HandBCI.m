
% init

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

parallel_check

%% load data
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
cd(root_path)
load session_data_B3_Hand
load('ECOG_Grid_8596_000067_B3.mat')

% load B3_waves_3DArrow_stability_hgFilterBank_PLV_AccStatsCL_v2
% num_targets=7;

load B3_waves_Hand_stability_hgFilterBank_PLV_AccStatsCL_v2_PLVDelta
num_targets=12;



%% WHAT HAPPENS TO TRAVELING WAVE PHENOMENA ACROSS DAYS IN B3 HAND
% MU POWER BETWEEN WAVES AND NON WAVES
% PAC BETWEEN WAVES AND NON WAVES

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
load B3_waves_Hand_stability_hgFilterBank_PLV_AccStatsCL_v2_PLVDelta

% does duty cycle change across days
dcyc_days_corr=[];
dcyc_days_err=[];
acc_days=[];
for i=1:length(stats_cl_days)
    a = stats_cl_days{i};
    l = round(length(a)/2);
    a = a(l+1:end);
    %a = a(end-24:end);
    dcyc_corr=[];
    dcyc_err=[];
    acc_days(i) = mean([a(1:end).accuracy]);
    for j=1:length(a)
        tmp  = a(j).stab;
        % correct trials
        if a(j).accuracy==1 
            tmp = tmp(1:end);
            [out,st,stp] = wave_stability_detect(zscore(tmp));
            t = length(tmp) * 20/1e3;
            f = length(out)/t; % frequency/s
            d = mean(out) * 20/1e3; %duration in s
            dcyc_corr=[dcyc_corr f*d];
        end

        % incorrect trials
        if a(j).accuracy==0
            tmp = tmp(1:end);
            [out,st,stp] = wave_stability_detect(zscore(tmp));
            t = length(tmp) * 20/1e3;
            f = length(out)/t; % frequency/s
            d = mean(out) * 20/1e3; %duration in s
            dcyc_err=[dcyc_err f*d];
        end
    end
    dcyc_days_corr(i) = mean(dcyc_corr);
    dcyc_days_err(i) = mean(dcyc_err);
end

figure;
boxplot([dcyc_days_corr'  dcyc_days_err'])
figure;
plot(dcyc_days_corr-dcyc_days_err,'.','MarkerSize',20)
tmp = dcyc_days_corr - dcyc_days_err;
day = (1:length(tmp))';
mdl = fitlm(day,tmp,'RobustOpts','on')
bhat = mdl.Coefficients.Estimate;
xhat = linspace(1,10,100);
yhat = predict(mdl,xhat(:));


%%%% does mu PAC difference between wave and nonwave epochs change across days
[res_days_B3] = get_plv_waves(stats_cl_hg_days,ecog_grid,cortex,elecmatrix);
[p,h,stats] = signrank(res_days_B3(:,1),res_days_B3(:,2),'method','approximate');
figure;
boxplot(res_days_B3);
xticks(1:2)
xticklabels({'Waves','Nonwaves'})
ylabel('Mu-hG PAC')
plot_beautify

%tmp = (res_days_B3(:,1) - res_days_B3(:,2))./(res_days_B3(:,2));
tmp = (res_days_B3(:,1) - res_days_B3(:,2));
%tmp = tmp(:)*100;
day = (1:length(tmp))';
mdl = fitlm(day,tmp,'RobustOpts','on')
bhat = mdl.Coefficients.Estimate;
xhat = linspace(1,10,100);
yhat = predict(mdl,xhat(:));

figure;plot(tmp,'.','MarkerSize',30)
hold on
plot(xhat,yhat,'k','LineWidth',1);
xticks(1:10)
xlabel('Days')
ylabel('Difference in PAC b/w Waves and Non-waves')
xlim([0.5 10.5])
plot_beautify

% slope and stats
beta  = mdl.Coefficients.Estimate(2);
SE    = mdl.Coefficients.SE(2);
tstat = mdl.Coefficients.tStat(2);
pval  = mdl.Coefficients.pValue(2);
fprintf('slope = %.4f +/- %.4f SE, t = %.3f, p = %.4g\n', ...
    beta,SE,tstat,pval);



%%%% does mu power difference between wave and nonwave epochs change across days
mu_wave_pow_days=[];
mu_nonwave_pow_days=[];
for i=1:length(stats_cl_hg_days)
    a = stats_cl_hg_days{i};
    l = round(length(a)/2);
    a = a(l+1:end);
    wave_pow=[];
    nonwave_pow=[];
    for j=1:length(a)
        tmp = abs(cell2mat(a(j).mu_wave'));
        wave_pow = [wave_pow mean(tmp(:))];

        tmp = abs(cell2mat(a(j).mu_nonwave'));
        nonwave_pow = [nonwave_pow mean(tmp(:))];
    end

    mu_wave_pow_days(i) = median(wave_pow);
    mu_nonwave_pow_days(i) = median(nonwave_pow);

end

figure;
boxplot([mu_wave_pow_days' mu_nonwave_pow_days'],'Whisker',1.6)
xticks(1:2)
xticklabels({'Waves','Nonwaves'})
ylabel('Mu Power (z)')
plot_beautify
[p,h,stats] = signrank(mu_wave_pow_days,mu_nonwave_pow_days,'method','approximate');


% 
% figure;
% plot(mu_wave_pow_days - mu_nonwave_pow_days,'.','MarkerSize',30)

tmp = (mu_wave_pow_days - mu_nonwave_pow_days);
%tmp = tmp(:)*100;
day = (1:length(tmp))';
mdl = fitlm(day,tmp,'RobustOpts','on')
bhat = mdl.Coefficients.Estimate;
xhat = linspace(1,10,100);
yhat = predict(mdl,xhat(:));

figure;plot(tmp,'.','MarkerSize',30)
hold on
plot(xhat,yhat,'k','LineWidth',1);
xticks(1:10)
xlabel('Days')
ylabel('Delta Mu Power  Waves - NonWaves')
xlim([0.5 10.5])
plot_beautify
ylim([0.04 0.12])




%% PLOTTING AS SVG


cd('C:\Users\nikic\Documents\Ganguly lab\ECoG BCI\BCI_Paper_Waves_Hand\Paper_text\Figures_New\Figure5\')

% svg
set(gcf,'PaperPositionMode','auto');
print(gcf,'WaveDutyCycle.svg','-dsvg','-painters','-r300');
    
% png
set(gcf,'PaperPositionMode','auto');
print(gcf,'B3_BrainElec.png','-dpng','-r500');

