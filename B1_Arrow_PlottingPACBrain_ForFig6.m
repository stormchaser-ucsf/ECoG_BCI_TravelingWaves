

clc;
clear;

addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))
cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves')
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')

imaging_B1_253 % already in blackrock sorted grid numbering
close all

lfo_cl_all=[];
mu_cl_all=[];
subj_idx=[];

%b1
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
b1_lfo = load('PAC_B1_LFO_hG_rawValues_New_v2.mat');
b1_mu = load('PAC_B1_Mu_hG_rawValues_New.mat');
% lfo sig ch
cl_days=[2:2:length(b1_lfo.pac_raw_values)];
pac_all_lfo=[];
cl=[];
sig_ch_lfo_days=[];
for i=1:length(cl_days)
    tmp = b1_lfo.pac_raw_values(cl_days(i)).pac;
    tmp = abs(mean(tmp));
    pac_all_lfo(i,:) = tmp;

    ptmp=b1_lfo.pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);    
    %pfdr = 0.013;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
    sig_ch_lfo_days(i,:) =  ptmp<=pfdr;
end
lfo_cl=cl;

% mu sig ch
cl_days=[2:2:length(b1_mu.pac_raw_values)];
pac_all_mu=[];
cl=[];
sig_ch_mu_days=[];
for i=1:length(cl_days)
    tmp = b1_mu.pac_raw_values(cl_days(i)).pac;
    tmp = abs(mean(tmp));
    pac_all_mu(i,:) = tmp;

    ptmp=b1_mu.pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);    
    %pfdr = 0.013;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
    sig_ch_mu_days(i,:) = ptmp<=pfdr;
end
mu_cl=cl;
figure;
boxplot([mu_cl' lfo_cl(1:end)'],'Symbol','')
lfo_cl_all = [lfo_cl_all;lfo_cl(1:end)'];
mu_cl_all = [mu_cl_all;mu_cl(1:end)'];
subj_idx = [subj_idx;ones(size(mu_cl,2),1)];
[p,h]=signrank(mu_cl,lfo_cl);
title(['B1 pval ' num2str(p)])
plot_beautify
xticks(1:2)
xticklabels({'Mu-hG','LFO-hG'})
ylabel('% Sig. Channels')
yticks([0:.1:1])
ylim([0 0.4])


% plotting on brain significant channels
% lfo
days=3:4;
sig_ch_lfo = sig_ch_lfo_days(days,:);
sig_ch_lfo = sum(sig_ch_lfo,1);
sig_ch_lfo(sig_ch_lfo>0)=1;
r_lfo = pac_all_lfo(days,:);
r_lfo = mean(r_lfo,1);
r_lfo(sig_ch_lfo==0)=0;
r_lfo =  r_lfo./max(r_lfo);
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix, 'color', 'w','msize',1);
for j=1:253
    if sig_ch_lfo(j)==1 && r_lfo(j)~=0
        ms = (r_lfo(j))*10;
        c='b';                
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end
plot_beautify
ax = gca;
% Center of electrode grid
grid_center = mean(elecmatrix, 1);
% Current camera position and target
old_target = camtarget(ax);
old_pos = campos(ax);
% Translate camera to electrode grid while preserving view direction
shift = grid_center - old_target;
campos(ax, old_pos + shift);
camtarget(ax, grid_center);
% Zoom in
camproj(ax, 'orthographic');
camzoom(ax, 2.4);
axis(ax, 'vis3d');
axis(ax, 'off');


% plotting on brain significant channels
% mu
sig_ch_mu = sig_ch_mu_days(days,:);
sig_ch_mu = sum(sig_ch_mu,1);
sig_ch_mu(sig_ch_mu>0)=1;
r_mu = pac_all_mu(days,:);
r_mu = mean(r_mu,1);
r_mu(sig_ch_mu==0)=0;
r_mu =  r_mu./max(r_mu);
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix, 'color', 'w','msize',1);
for j=1:253
    if sig_ch_mu(j)==1 && r_mu(j)~=0
        ms = (r_mu(j))*7;
        c='b';                
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end
plot_beautify
ax = gca;
% Center of electrode grid
grid_center = mean(elecmatrix, 1);
% Current camera position and target
old_target = camtarget(ax);
old_pos = campos(ax);
% Translate camera to electrode grid while preserving view direction
shift = grid_center - old_target;
campos(ax, old_pos + shift);
camtarget(ax, grid_center);
% Zoom in
camproj(ax, 'orthographic');
camzoom(ax, 2.4);
axis(ax, 'vis3d');
axis(ax, 'off');



