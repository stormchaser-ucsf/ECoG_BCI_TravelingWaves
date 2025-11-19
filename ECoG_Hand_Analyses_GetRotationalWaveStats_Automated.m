%ECoG_Hand_Analyses_GetRotationalWaveStats_Automated
% Goal is to get rotatinal wave stats and contrast OL and CL across all
% channels of the CNN AE in an automated manner


%% init
clc;clear

if ispc


    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
    cd(root_path)
    load('ECOG_Grid_8596_000067_B3.mat')

    % add the circ stats toolbox
    addpath('C:\Users\nikic\Documents\MATLAB')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\wave-matlab-master\wave-matlab-master'))
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves'))

    %imaging_B3_waves;close all

    folderpath='C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\CNN3D\Eigmaps_B1';
    files = findfiles('',folderpath)'; % go by Layer ->  Channels


else


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

end

%% MANUAL comparing the max curl value between OL and CL across all layers

res=[];num_days=9;
for iter=1:length(files)
    load(files{iter})
    max_curl_OL=[];
    max_curl_CL=[];
    res_pc=[];
    for pc_idx=1:5
        ol=[];
        cl=[];
        for j=pc_idx:5:num_days*5
            xph_OL = squeeze(OL(j,:,:));
            curl_OL =  get_curl(xph_OL);
            tmp=max(abs(curl_OL(:)));
            % if tmp<=0.35
            %     tmp=NaN;
            % end
            ol = [ol tmp];
            xph_CL = squeeze(CL(j,:,:));
            curl_CL =  get_curl(xph_CL);
            tmp = max(abs(curl_CL(:)));
            % if tmp<=0.35
            %     tmp=NaN;
            % end
            cl = [cl tmp];
        end
        if  sum(isnan(cl)) > 5 ||  sum(isnan(ol)) > 5
            res_pc(pc_idx) = 0;
        else
            [p,h]=signrank((cl),(ol));
            if median(cl-ol)>0
                %[h,p,tb,stats]=ttest(cl,ol);
                res_pc(pc_idx) = h;
            else
                res_pc(pc_idx) = 0;
            end
        end
    end
    res(:,iter) = res_pc;
end
figure;imagesc(res)

% now looking at the gradient fields individually
iter = 9;
load(files{iter})
pc_idx=3;
figure;
ha=tight_subplot(5,2);
k=1;
for j=pc_idx:5:num_days*5     
    xph = squeeze(CL(j,:,:));
    [XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );
    % smooth phasors to get smoothed estimates of phase
    M = real(xph);
    N = imag(xph);
    M = smoothn(M,'robust'); %dx
    N = smoothn(N,'robust'); %dy
    xphs = M + 1j*N; % smoothed phasor field: gets smoothed estimates of phase

    % compute gradient
    [pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN( xphs, ...
        1,-1);

    M =  1.*cos(pd);
    N =  1.*sin(pd);

    [curl_val] = curl(XX,YY,M,N);
    [div_val] = divergence(XX,YY,M,N);

    axes(ha(k))
    quiver(XX,YY,M,N)    
    axis tight
    k=k+1;
end


%% automated detection of curl regions

% get across all layers
for iter = 1:length(files)
    load(files{iter})    
    max_curl_OL=[];
    max_curl_CL=[];
    for j=3:5:45
        xph_CL = squeeze(CL(j,:,:));
        xph_OL = squeeze(OL(j,:,:));
        out_OL = curl_stats(xph_OL);
        out_CL = curl_stats(xph_CL);
        corr_cl=[];
        for k=1:length(out_CL)
            tmp =  out_CL(k).corr;
            if isempty(tmp)
                tmp=NaN;
            end
            corr_cl(k) = tmp;
        end
        corr_ol=[];
        for k=1:length(out_OL)
            tmp =  out_OL(k).corr;
            if isempty(tmp)
                tmp=NaN;
            end
            corr_ol(k) = tmp;
        end

        max_curl_OL = [max_curl_OL nanmean(corr_ol)];
        max_curl_CL = [max_curl_CL nanmean(corr_cl)];                      
       
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
