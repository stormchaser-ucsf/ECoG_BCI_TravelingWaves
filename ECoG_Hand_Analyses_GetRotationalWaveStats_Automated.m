%ECoG_Hand_Analyses_GetRotationalWaveStats_Automated
% Goal is to get rotatinal wave stats and contrast OL and CL across all
% channels of the CNN AE in an automated manner


%% init
clc;clear
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3';
cd(root_path)
load('ECOG_Grid_8596_000067_B3.mat')

% add the circ stats toolbox
addpath('/home/user/Documents/MATLAB')
addpath('/home/user/Documents/MATLAB/CircStat2012a')
addpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/helpers')
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/wave-matlab-master/wave-matlab-master'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves'))
imaging_B3_waves;close all

folderpath='/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/CNN3D/Eigmaps';
files = findfiles('',folderpath)'; % go by Layer ->  Channels


%% 

% get across all layers
for iter = 1:length(files)
    load(files{iter})
    tmp_data=[];
    max_curl_OL=[];
    max_curl_CL=[];
    for j=3:5:50
        xph_CL = squeeze(CL(j,:,:));
        xph_OL = squeeze(OL(j,:,:));
        out_OL=curl_stats(xph_OL);
        out_CL=curl_stats(xph_CL);

        max_curl_OL = [max_curl_OL out_OL.cc1];
        max_curl_CL = [max_curl_CL out_CL.cc1];
                      
        % if max(abs(curl_val0(:)))>=0.75
        %     tmp_data=cat(3,tmp_data,curl_val);
        % end

      
        % map onto electrodes
        %wts = map_onto_elec_B3(abs(curl_val),ecog_grid,grid_layout);
        % plot on brain
        %subplot(5,2,ii)
        %axes(ha(ii))

        %c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
        %wts1=wts./scale_factor;
        %for i=1:length(wts1)
        %    m = wts1(i)*10;
        %    e_h = el_add(elecmatrix(i,:),'color','b', 'msize',m);
        %end
        %title(['Day ' num2str(ii)])

        %ii=ii+1;
        %tmp_data=[tmp_data wts];
    end
    figure;boxplot(abs(max_curl_CL)-abs(max_curl_OL))
    hline(0)
    [p,h]=signrank(abs(max_curl_CL),abs(max_curl_OL));
    title(['pval of ' num2str(p)])


    wts = nanmean(tmp_data,3);
    if isempty(wts)
        wts=1e-6*ones(11,23);
    end
    wts = map_onto_elec_B3(abs(wts),ecog_grid,grid_layout);
    figure
    c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
    scale_factor = max(wts(:));
    wts1=wts./scale_factor;
    for i=1:length(wts1)
        m = wts1(i)*10;
        e_h = el_add(elecmatrix(i,:),'color','b', 'msize',m);
    end
    title(files{iter}(end-15:end-7));
end
