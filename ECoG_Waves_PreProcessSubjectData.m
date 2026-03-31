%% WAVE PROCESSING SUBJECTS DATA
clear
clc
close all
subj='B3';
%% LOAD SUBJECT SPECIFIC DATA

if strcmp(subj,'B3')

    if ispc
        root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
        addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
        cd(root_path)
        addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
        %load session_data_B3_Hand
        load session_data_B3
        addpath 'C:\Users\nikic\Documents\MATLAB'
        load('ECOG_Grid_8596_000067_B3.mat')
        addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\'))

    else
        %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
        root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
        cd(root_path)
        %load session_data_B3_Hand
        load session_data_B3
        load('ECOG_Grid_8596_000067_B3.mat')
        addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))

    end


    d1 = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
        'SampleRate',1e3);
    d2 = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
        'SampleRate',50);
    bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
        'SampleRate',1e3);

    hilbert_flag=1;


    imaging_B3_waves;
    len_days = min(11,length(session_data));
    num_targets=7;
end


if strcmp(subj,'B1')

    if ispc
        root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
        addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
        cd(root_path)
        addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
        %load session_data_B3_Hand
        addpath 'C:\Users\nikic\Documents\MATLAB'
        load('ECOG_Grid_8596_000067_B3.mat')
        addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\'))

    else
        %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
        root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
        cd(root_path)
        %load session_data_B3_Hand
        load('ECOG_Grid_8596_000067_B3.mat')
        addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))

    end


    % was earlier 7 to 9
    d1 = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
        'SampleRate',1e3); % center freq is 8.5
    d2 = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
        'SampleRate',50);
    bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
        'SampleRate',1e3); % center freq is 110
    hilbert_flag=1;


    imaging_B3_waves;

    folders={'20240515', '20240517', '20240614', ...
        '20240619', '20240621', '20240626',...
        '20240710','20240712','20240731'};
     num_targets=7;
end

if strcmp(subj,'B6')
    if ispc
        root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B6';
        addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
        cd(root_path)
        addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
        addpath 'C:\Users\nikic\Documents\MATLAB'
        load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\ECOG_Grid_8596_000067_B3.mat')
        addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers'))

    else
        %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
        root_path='/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
        cd(root_path)
        %load session_data_B3_Hand
        load('ECOG_Grid_8596_000067_B3.mat')
        addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))

    end



    % seems to be between 7 and 10Hz for arrow
    d1 = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',7.0,'HalfPowerFrequency2',10, ...
        'SampleRate',1e3);

    d2 = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',7.0,'HalfPowerFrequency2',10, ...
        'SampleRate',50);

    d3 = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
        'SampleRate',1e3);

    bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
        'SampleRate',1e3);

    deltaFilt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
        'SampleRate',50);

    bpFilt=deltaFilt;



    %
    % % hand
    % folders={'20250624', '20250703', ...
    %      '20250827', '20250903', '20250917','20250924'}; %20250708 has only imagined

    % robot3DArrow
    folders = {'20250530','20250610','20250624','20250703','20250708','20250717',...
        '20250917','20250924','20251203','20251204','20251210','20260116'};
    %20250924 seems to have no closed loop data?  -> have to add that in folder
    %list above
    imaging_B3_waves;
end


%% MAIN CODE TO PROCESS B3 HAND/ARROW

% load cl2 trials from last day
%i=length(session_data);
xol_days=[];
xcl_days=[];
stats_ol_days={};
stats_cl_days={};
len_days = min(11,length(session_data));
stats_ol_hg_days={};
stats_cl_hg_days={};
wave_plv_ol_days={};
nonwave_plv_ol_days={};
wave_plv_cl_days={};
nonwave_plv_cl_days={};
for days=1:len_days

    disp(['Processing day ' num2str(days)])

    folders_imag =  strcmp(session_data(days).folder_type,'I');
    folders_online = strcmp(session_data(days).folder_type,'O');
    folders_batch = strcmp(session_data(days).folder_type,'B');
    folders_batch1 = strcmp(session_data(days).folder_type,'B1');
    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    batch_idx1 = find(folders_batch1==1);
    online_idx=[online_idx batch_idx batch_idx1];
    %online_idx=[online_idx batch_idx batch_idx1];
    %online_idx = [batch_idx batch_idx1];



    %%%%%% get imagined data files
    folders = session_data(days).folders(imag_idx);
    day_date = session_data(days).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        %folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    len = min(150,length(files));
    idx=randperm(length(files),len);
    [stats_ol,stats_ol_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
        grid_layout,elecmatrix,bpFilt,d1,0);
    stats_ol_days{days}=stats_ol;
    stats_ol_hg_days{days} = stats_ol_hg;

    % %  % stats on PLV, OL
    % wave_plv=[];nonwave_plv=[];wave_len=[];nonwave_len=[];nonwave_len_corr=[];
    % wave_angle=[];nonwave_angle=[];
    % for i=1:length(stats_ol_hg)
    %     %%% just straight up average plv across grid
    %     a=stats_ol_hg(i).plv_wave;
    %     wave_len(i) = size(a,1);
    %     wave_plv(i) = mean(abs(mean(a)));
    %     %wave_plv(i,:) = ((mean(a)));
    % 
    %     a = stats_ol_hg(i).plv_nonwave;
    %     nonwave_len(i) = size(a,1);
    %     if nonwave_len(i) > wave_len(i)
    %         idx = randperm(size(a,1),wave_len(i));
    %         %a = a(idx,:);
    %         a = a(1:wave_len(i),:);
    %     end          
    %     nonwave_len_corr(i) = size(a,1);
    %     nonwave_plv(i) = mean(abs(mean(a)));
    %     %nonwave_plv(i,:) = ((mean(a)));
    % 
    %     %%% based on phase consistency across trials
    %     % a=stats_ol_hg(i).plv_wave;
    %     % wave_angle(i,:)=angle(mean(a));
    %     % a=stats_ol_hg(i).plv_nonwave;
    %     % nonwave_angle(i,:)=angle(mean(a));
    % end
    % 
    % wave_plv =abs(mean(wave_plv,1));
    % nonwave_plv =abs(mean(nonwave_plv,1));

    % wave_plv_ol = wave_plv;
    % nonwave_plv_ol= nonwave_plv;
    % figure;boxplot([wave_plv' nonwave_plv']);
    % xticks(1:2)
    % xticklabels({'Wave epochs','Non wave epochs'})
    % [p,h]=signrank(wave_plv,nonwave_plv)
    % ylabel('Grid-wise PAC between high gamma and mu')


    %%%%%% get online data files %%%%%
    folders = session_data(days).folders(online_idx);
    day_date = session_data(days).Day;
    files=[];
    for ii=1:length(folders)
        %folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    len = min(150,length(files));
    idx=randperm(length(files),len);
     [stats_cl,stats_cl_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
        grid_layout,elecmatrix,bpFilt,d1,1);
    stats_cl_days{days}=stats_cl;
    stats_cl_hg_days{days}=stats_cl_hg;

    % looking at mu power during waves vs. non wave events
    mu_wave_pow=[];
    mu_nonwav_pow=[];
    for ii=1:length(stats_cl_hg)
        tmp = stats_cl_hg(ii).mu_wave;
        tmp = cell2mat(tmp');
        tmp = abs(tmp);
        mu_wave_pow(ii,:) = mean(tmp,1);

        tmp = stats_cl_hg(ii).mu_nonwave;
        tmp = cell2mat(tmp');
        tmp = abs(tmp);
        mu_nonwav_pow(ii,:) = mean(tmp,1);
    end

    figure;
    boxplot([mean(mu_wave_pow,1)' mean(mu_nonwav_pow,1)'])



    %   % stats on PLV, CL
    % wave_plv=[];nonwave_plv=[];wave_len=[];nonwave_len=[];nonwave_len_corr=[];
    % wave_angle=[];nonwave_angle=[];
    % wave_plv_trial=[];nonwave_plv_trial=[];
    % for i=1:length(stats_cl_hg)
    %     %%% just straight up average plv across grid
    %     a=stats_cl_hg(i).plv_wave;
    %     wave_len(i) = size(a,1);
    %     wave_plv(i) = mean(abs(mean(a)));
    %     wave_plv_trial(i,:) = abs(mean(a));
    %     %wave_plv(i,:) = ((mean(a)));
    % 
    %     a = stats_cl_hg(i).plv_nonwave;
    %     nonwave_len(i) = size(a,1);
    %     if nonwave_len(i) > wave_len(i)
    %         idx = randperm(size(a,1),wave_len(i));
    %         %a = a(idx,:);
    %         a = a(1:wave_len(i),:);
    %     end          
    %     nonwave_len_corr(i) = size(a,1);
    %     nonwave_plv(i) = mean(abs(mean(a)));
    %     nonwave_plv_trial(i,:) = abs(mean(a));
    %     %nonwave_plv(i,:) = ((mean(a)));
    % 
    %     %%% based on phase consistency across trials
    %     % a=stats_ol_hg(i).plv_wave;
    %     % wave_angle(i,:)=angle(mean(a));
    %     % a=stats_ol_hg(i).plv_nonwave;
    %     % nonwave_angle(i,:)=angle(mean(a));
    % end
    % % wave_plv =abs(mean(wave_plv,1));
    % % nonwave_plv =abs(mean(nonwave_plv,1));
    % 
    % wave_plv_cl = wave_plv;
    % nonwave_plv_cl= nonwave_plv;
    % figure;boxplot([wave_plv' nonwave_plv']);
    % xticks(1:2)
    % xticklabels({'Wave epochs','Non wave epochs'})
    % [p,h]=signrank(wave_plv,nonwave_plv)
    % ylabel('Grid-wise PAC between high gamma and mu')


    % % store
    % wave_plv_ol_days{days} = wave_plv_ol;
    % wave_plv_cl_days{days} = wave_plv_cl;
    % nonwave_plv_ol_days{days} = nonwave_plv_ol;
    % nonwave_plv_cl_days{days} = nonwave_plv_cl;
    % 
    % % plotting
    % if length(wave_plv_ol) < length(wave_plv_cl)
    %     wave_plv_ol(end+1:length(wave_plv_cl)) = NaN;
    %     nonwave_plv_ol(end+1:length(wave_plv_cl)) = NaN;
    % elseif length(wave_plv_ol) > length(wave_plv_cl)
    %     wave_plv_cl(end+1:length(wave_plv_ol)) = NaN;
    %     nonwave_plv_cl(end+1:length(wave_plv_ol)) = NaN;
    % end
    % figure;boxplot([wave_plv_ol' nonwave_plv_ol' wave_plv_cl' nonwave_plv_cl']);
    % xticks(1:4)
    % xticklabels({'OL Wave epochs','OL Non wave epochs',...
    %     'CL Wave epochs','CL Non wave epochs'})
    % ylabel('Grid-wise PAC between high gamma and mu')
    % plot_beautify
    % title(['Day ' num2str(days)])


    x=[];
    for i = 1:length(stats_ol)
        %tmp = stats_ol(i).size;
        %tmp = stats_ol(i).corr;
        %x(i,:) = smooth(tmp(1:2.2e3),50);
        %tmp = smooth(tmp,50);
        %x= [x; nanmedian(tmp)];
        %x=[x;sum(isnan(tmp))/length(tmp)];
        %x(i) = median(abs(tmp));
        tmp = (stats_ol(i).stab);
        tmp=zscore(tmp(1:end));
        %x(i) = sum(tmp>0.0)/length(tmp);
        out = wave_stability_detect(tmp);
        %x(i) = median(out);
        %x(i) = sum(out)/length(tmp);
        %x(i) = length(out)/length(tmp);
        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(i) = f*d;
    end
    %figure;plot(mean(x,1))
    xol=x;

    x=[];
    for i = 1:length(stats_cl)
        %tmp = stats_cl(i).corr;
        %tmp = smooth(tmp,10);
        %x= [x; nanmedian(tmp)];
        %x(i,:) = smooth(tmp(1:500),50);
        %x=[x;sum(isnan(tmp))/length(tmp)];
        %x(i) = median(abs(tmp));
        tmp = (stats_cl(i).stab);
        tmp=zscore(tmp(1:end));
        %x(i) = sum(tmp>0.0)/length(tmp);
        out = wave_stability_detect(tmp);
        %x(i) = median(out);
        %x(i) = sum(out)/length(tmp);
        %x(i) = length(out)/length(tmp);
        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(i) = f*d;
    end
    xcl=x;

    %xol(end+1:length(xcl))=NaN;
    %[p,h] = ranksum(xol,xcl)
    %figure;boxplot([xol' xcl'],'notch','off')
    %title(['Day ' num2str(days)]);
    xcl_days(days)=mean(xcl);
    xol_days(days)=mean(xol);

    %[p,h] = signrank(xol,xcl)
    %figure;boxplot([xol' xcl'],'notch','off')
    
end

%save B3_waves_hand_stability_Muller_hG -v7.3
%save B3_waves_hand_stability_Muller_hG_plv -v7.3
save B3_waves_Hand_stability_hgFilterBank_PLV_AccStatsCL_v2_PLVDelta -v7.3

% (IMPT) v2 is hg_smoothed by 10 samples (200ms)

%% MAIN CODE TO PROCESS B1 and B6


hilbert_flag=1;
xol_days=[];
xcl_days=[];
stats_ol_days={};
stats_cl_days={};
stats_ol_hg_days={};
stats_cl_hg_days={};
wave_plv_ol_days={};
nonwave_plv_ol_days={};
wave_plv_cl_days={};
nonwave_plv_cl_days={};
for days=1:length(folders) %if B1-> it is -1

    disp(['Processing day ' num2str(days)])

    folderpath = fullfile(root_path,folders{days},'Robot3DArrow');
    % if i<=2
    %     folderpath = fullfile(root_path,folders_robot{i},'Robot3D');
    % else
    %     folderpath = fullfile(root_path,folders_robot{i},'RealRobotBatch');
    % end
    D= dir(folderpath);
    D = D(3:end);
    imag_idx=[];
    online_idx=[];
    for j=1:length(D)
        subfoldername = dir(fullfile(folderpath,D(j).name));
        if length(subfoldername)>2
            if strcmp(subfoldername(3).name,'Imagined')
                imag_idx=[imag_idx j];
            elseif strcmp(subfoldername(3).name,'BCI_Fixed')
                online_idx=[online_idx j];
            end
        end
    end
    % imag_idx_main=imag_idx(1:3);
    % online_idx_main=online_idx(1:3);
    %
    % folders_imag =  strcmp(session_data(days).folder_type,'I');
    % folders_online = strcmp(session_data(days).folder_type,'O');
    % folders_batch = strcmp(session_data(days).folder_type,'B');
    % folders_batch1 = strcmp(session_data(days).folder_type,'B1');
    % imag_idx = find(folders_imag==1);
    % online_idx = find(folders_online==1);
    % batch_idx = find(folders_batch==1);
    % batch_idx1 = find(folders_batch1==1);
    % online_idx=[online_idx batch_idx];
    % online_idx=[online_idx batch_idx batch_idx1];
    %online_idx = [batch_idx batch_idx1];



    %%%%%% get imagined data files
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    len = min(200,length(files));
    idx=randperm(length(files),len);
    % [stats_ol,stats_ol_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
    %      grid_layout,elecmatrix,bpFilt,d1,0);
    stats_ol=[];stats_ol_hg=[];
    stats_ol_days{days}=stats_ol;
    stats_ol_hg_days{days} = stats_ol_hg;

    % % stats on PLV, OL
    % wave_plv=[];nonwave_plv=[];wave_len=[];nonwave_len=[];nonwave_len_corr=[];
    % wave_angle=[];nonwave_angle=[];
    % for i=1:length(stats_ol_hg)
    %     %%% just straight up average plv across grid
    %     a=stats_ol_hg(i).plv_wave;
    %     wave_len(i) = size(a,1);
    %     wave_plv(i) = mean(abs(mean(a)));
    %     %wave_plv(i,:) = ((mean(a)));
    % 
    %     a = stats_ol_hg(i).plv_nonwave;
    %     nonwave_len(i) = size(a,1);
    %     if nonwave_len(i) > wave_len(i)
    %         idx = randperm(size(a,1),wave_len(i));
    %         a = a(idx,:);
    %     end
    %     nonwave_len_corr(i) = size(a,1);
    %     nonwave_plv(i) = mean(abs(mean(a)));
    %     %nonwave_plv(i,:) = ((mean(a)));
    % 
    %     %%% based on phase consistency across trials
    %     % a=stats_ol_hg(i).plv_wave;
    %     % wave_angle(i,:)=angle(mean(a));
    %     % a=stats_ol_hg(i).plv_nonwave;
    %     % nonwave_angle(i,:)=angle(mean(a));
    % end
    % 
    % % wave_plv =abs(mean(wave_plv,1));
    % % nonwave_plv =abs(mean(nonwave_plv,1));
    % 
    % wave_plv_ol = wave_plv;
    % nonwave_plv_ol= nonwave_plv;
    % figure;boxplot([wave_plv' nonwave_plv']);
    % xticks(1:2)
    % xticklabels({'Wave epochs','Non wave epochs'})
    % [p,h]=signrank(wave_plv,nonwave_plv)
    % ylabel('Grid-wise PAC between high gamma



    % wave_angle = mean(exp(1i.*wave_angle),1);
    % nonwave_angle = mean(exp(1i.*nonwave_angle),1);





    %%%%%% get online data files %%%%%
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    len = min(200,length(files));
    idx=randperm(length(files),len);
    [stats_cl,stats_cl_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
        grid_layout,elecmatrix,bpFilt,d1,1);
    stats_cl_days{days}=stats_cl;
    stats_cl_hg_days{days}=stats_cl_hg;


    % looking at mu power during waves vs. non wave events
    mu_wave_pow=[];
    mu_nonwav_pow=[];
    for ii=1:length(stats_cl_hg)
        tmp = stats_cl_hg(ii).mu_wave;
        tmp = cell2mat(tmp');
        tmp = abs(tmp);
        mu_wave_pow(ii,:) = mean(tmp,1);

        tmp = stats_cl_hg(ii).mu_nonwave;
        tmp = cell2mat(tmp');
        tmp = abs(tmp);
        mu_nonwav_pow(ii,:) = mean(tmp,1);
    end

    figure;
    boxplot([mean(mu_wave_pow,1)' mean(mu_nonwav_pow,1)'])

   

    % stats on PLV, CL
    % wave_plv=[];nonwave_plv=[];wave_len=[];nonwave_len=[];nonwave_len_corr=[];
    % wave_angle=[];nonwave_angle=[];
    % wave_plv_trial=[];nonwave_plv_trial=[];
    % for i=1:length(stats_cl_hg)
    %     %%% just straight up average plv across grid
    %     a=stats_cl_hg(i).plv_wave;
    %     wave_len(i) = size(a,1);
    %     wave_plv(i) = mean(abs(mean(a)));
    %     wave_plv_trial(i,:) = abs(mean(a));
    %     %wave_plv(i,:) = ((mean(a)));
    % 
    %     a = stats_cl_hg(i).plv_nonwave;
    %     nonwave_len(i) = size(a,1);
    %     if nonwave_len(i) > wave_len(i)
    %         idx = randperm(size(a,1),wave_len(i));
    %         a = a(idx,:);
    %     end
    %     nonwave_len_corr(i) = size(a,1);
    %     nonwave_plv(i) = mean(abs(mean(a)));
    %     nonwave_plv_trial(i,:) = abs(mean(a));
    %     %nonwave_plv(i,:) = ((mean(a)));
    % 
    %     %%% based on phase consistency across trials
    %     % a=stats_ol_hg(i).plv_wave;
    %     % wave_angle(i,:)=angle(mean(a));
    %     % a=stats_ol_hg(i).plv_nonwave;
    %     % nonwave_angle(i,:)=angle(mean(a));
    % end
    % % wave_plv =abs(mean(wave_plv,1));
    % % nonwave_plv =abs(mean(nonwave_plv,1));
    % 
    % wave_plv_cl = wave_plv;
    % nonwave_plv_cl= nonwave_plv;
    % % figure;boxplot([wave_plv' nonwave_plv']);
    % % xticks(1:2)
    % % xticklabels({'Wave epochs','Non wave epochs'})
    % % [p,h]=signrank(wave_plv,nonwave_plv)
    % % ylabel('Grid-wise PAC between high gamma and mu')
    % 
    % 
    % % store
    % wave_plv_ol_days{days} = wave_plv_ol;
    % wave_plv_cl_days{days} = wave_plv_cl;
    % nonwave_plv_ol_days{days} = nonwave_plv_ol;
    % nonwave_plv_cl_days{days} = nonwave_plv_cl;

    % % plotting
    % if length(wave_plv_ol) < length(wave_plv_cl)
    %     wave_plv_ol(end+1:length(wave_plv_cl)) = NaN;
    %     nonwave_plv_ol(end+1:length(wave_plv_cl)) = NaN;
    % elseif length(wave_plv_ol) > length(wave_plv_cl)
    %     wave_plv_cl(end+1:length(wave_plv_ol)) = NaN;
    %     nonwave_plv_cl(end+1:length(wave_plv_ol)) = NaN;
    % end
    % figure;boxplot([wave_plv_ol' nonwave_plv_ol' wave_plv_cl' nonwave_plv_cl']);
    % xticks(1:4)
    % xticklabels({'OL Wave epochs','OL Non wave epochs',...
    %     'CL Wave epochs','CL Non wave epochs'})
    % ylabel('Grid-wise PAC between high gamma and mu')
    % plot_beautify
    % title(['Day ' num2str(days)])




    x=[];
    for i = 1:length(stats_ol)
        %tmp = stats_ol(i).size;
        %tmp = stats_ol(i).corr;
        %x(i,:) = smooth(tmp(1:2.2e3),50);
        %tmp = smooth(tmp,50);
        %x= [x; nanmedian(tmp)];
        %x=[x;sum(isnan(tmp))/length(tmp)];
        %x(i) = median(abs(tmp));
        tmp = (stats_ol(i).stab);
        tmp=zscore(tmp(1:end));
        %x(i) = sum(tmp>0.0)/length(tmp);
        [out,st,stp] = wave_stability_detect(tmp);
        %x(i) = median(out);
        %x(i) = sum(out)/length(tmp);
        %x(i) = length(out)/length(tmp);
        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(i) = f*d;

        % stability values
    end
    %figure;plot(mean(x,1))
    xol=x;

    x=[];
    for i = 1:length(stats_cl)
        %tmp = stats_cl(i).corr;
        %tmp = smooth(tmp,10);
        %x= [x; nanmedian(tmp)];
        %x(i,:) = smooth(tmp(1:500),50);
        %x=[x;sum(isnan(tmp))/length(tmp)];
        %x(i) = median(abs(tmp));
        tmp = (stats_cl(i).stab);
        tmp=zscore(tmp(1:end));
        %x(i) = sum(tmp>0.0)/length(tmp);
        out = wave_stability_detect(tmp);
        %x(i) = median(out);
        %x(i) = sum(out)/length(tmp);
        %x(i) = length(out)/length(tmp);
        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(i) = f*d;
    end
    xcl=x;

    %xol(end+1:length(xcl))=NaN;
    %[p,h] = ranksum(xol,xcl)
    %figure;boxplot([xol' xcl'],'notch','off')
    %title(['Day ' num2str(days)]);
    xcl_days(days)=mean(xcl);
    xol_days(days)=mean(xol);

    %save B6_waves_stability -v7.3 % 50Hz, removing last 400ms in fitering step

end

%save B1_waves_stability_hg_PLV_AccStatsCL -v7.3 %
%save B1_waves_stability_hgFilterBank_PLV_AccStatsCL_v2 -v7.3 %
%save B6_waves_stability_hgFilterBank_PLV_AccStatsCL_v2_AllData -v7.3 %

save B6_waves_stability_hgFilterBank_PLV_AccStatsCL_v2_AllData_PLVDetla -v7.3


%% LOOKING AT MU POWER IN EACH STATE ACROSS DAYS
% B3 HAND/ARROW

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

state_pow_days_ol={};
state_pow_days_cl={};
for days=1:len_days

    disp(['Processing day ' num2str(days)])

    folders_imag =  strcmp(session_data(days).folder_type,'I');
    folders_online = strcmp(session_data(days).folder_type,'O');
    folders_batch = strcmp(session_data(days).folder_type,'B');
    folders_batch1 = strcmp(session_data(days).folder_type,'B1');
    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    batch_idx1 = find(folders_batch1==1);
    online_idx=[online_idx batch_idx batch_idx1];
    %online_idx=[online_idx batch_idx batch_idx1];
    %online_idx = [batch_idx batch_idx1];



    %%%%%% get imagined data files
    folders = session_data(days).folders(imag_idx);
    day_date = session_data(days).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        %folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    state_pow = get_state_pow(files,bpFilt);
    %title(['Day ' num2str(days) ' OL'])
    state_pow_days_ol{days}=state_pow;
    


    %%%%%% get online data files %%%%%
    folders = session_data(days).folders(online_idx);
    day_date = session_data(days).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end


    state_pow = get_state_pow(files,bpFilt);
    %title(['Day ' num2str(days) ' CL'])
    state_pow_days_cl{days}=state_pow;

   
end


% 
% tmp=state_pow(:,2)';
% tmp1 = [tmp(1:107) 0 tmp(108:111) 0  tmp(112:115) 0 ...
%     tmp(116:end)];
% 
% figure;imagesc(tmp1(ecog_grid))

days=1:10;
pow=[];
for i=1:10
    tmp = state_pow_days_cl{i};
    pow(:,i) = tmp(:,3);
end
figure;
boxplot(pow)
ylabel('Z-score')
title('Mu power during BCI control')
xlabel('Days')
xticks(1:10)
plot_beautify
hline(0)

% splitting early vs late
early_pow = pow(:,1:5);
late_pow = pow(:,6:end);
figure;
boxplot([early_pow(:) late_pow(:)],'Notch','on')
ylabel('Z-score')
title('Mu power during BCI control (All chan)')
xticks(1:2)
xticklabels({'1st 5 Days','2nd 5 Days'})
plot_beautify
hline(0)


% plot on brain
tmp = (mean(late_pow,2) - mean(early_pow,2))';
tmp1 = [tmp(1:107) 0 tmp(108:111) 0  tmp(112:115) 0 ...
    tmp(116:end)];
figure;
imagesc(tmp1(ecog_grid))


%% LOOKING AT MU POWER IN EACH STATE ACROSS DAYS
% B1/B6 ARROW

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

state_pow_days_ol={};
state_pow_days_cl={};
for days=1:len_days

    disp(['Processing day ' num2str(days)])

    folderpath = fullfile(root_path,folders{days},'Robot3DArrow');
    % if i<=2
    %     folderpath = fullfile(root_path,folders_robot{i},'Robot3D');
    % else
    %     folderpath = fullfile(root_path,folders_robot{i},'RealRobotBatch');
    % end
    D= dir(folderpath);
    D = D(3:end);
    imag_idx=[];
    online_idx=[];
    for j=1:length(D)
        subfoldername = dir(fullfile(folderpath,D(j).name));
        if length(subfoldername)>2
            if strcmp(subfoldername(3).name,'Imagined')
                imag_idx=[imag_idx j];
            elseif strcmp(subfoldername(3).name,'BCI_Fixed')
                online_idx=[online_idx j];
            end
        end
    end



    %%%%%% get imagined data files
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    state_pow = get_state_pow(files,bpFilt);
    title(['Day ' num2str(days) ' OL'])
    state_pow_days_ol{days}=state_pow;
    


    %%%%%% get online data files %%%%%
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end


    state_pow = get_state_pow(files,bpFilt);
    title(['Day ' num2str(days) ' CL'])
    state_pow_days_cl{days}=state_pow;

   
end


% 
tmp=state_pow(:,2)';
tmp1 = [tmp(1:107) 0 tmp(108:111) 0  tmp(112:115) 0 ...
    tmp(116:end)];

figure;imagesc(tmp1(ecog_grid))



%% (MAIN) GETTING PAC BETWEEN MU AND HG IN ARROW TASK
% B1,B6

% 
% d1 = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
%     'SampleRate',1e3); % 8 to 10 or 0.5 to 5



d1 = designfilt('lowpassiir', 'FilterOrder', 4, ...
               'HalfPowerFrequency', 3, 'SampleRate', 1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

pac_ol=[];pval_ol=[];
pac_cl=[];pval_cl=[];
pac_batch=[];pval_batch=[];
rboot_ol=[];rboot_cl=[];rboot_batch=[];
pac_raw_values={};k=1;
tic
for i=1:length(folders)-1%go up to 8


    days=i;
    disp(['Processing day ' num2str(days)])

    folderpath = fullfile(root_path,folders{days},'Robot3DArrow');
    % if i<=2
    %     folderpath = fullfile(root_path,folders_robot{i},'Robot3D');
    % else
    %     folderpath = fullfile(root_path,folders_robot{i},'RealRobotBatch');
    % end
    D= dir(folderpath);
    D = D(3:end);
    imag_idx=[];
    online_idx=[];
    for j=1:length(D)
        subfoldername = dir(fullfile(folderpath,D(j).name));
        if length(subfoldername)>2
            if strcmp(subfoldername(3).name,'Imagined')
                imag_idx=[imag_idx j];
            elseif strcmp(subfoldername(3).name,'BCI_Fixed')
                online_idx=[online_idx j];
            end
        end
    end



    %%%%%% get imagined data files
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    len = min(80,length(files));
    idx=randperm(length(files),len);
    files=files(idx);

    % get the phase locking value
    if length(files)>0
        disp(['Processing Day ' num2str(i) ' OL'])
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    else
        pac=[];
    end


    % run permutation test and get pvalue for each channel
    %[pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    pval=[];
    rboot=[];

    %sum(pac_r>0.3)/253
    %pval_ol(i,:) = pval;
    %pac_ol(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'OL';
    pac_raw_values(k).Day = i;
    k=k+1;


    %%%%%% get online data files %%%%%
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    len = min(80,length(files));
    idx=randperm(length(files),len);
    files=files(idx);

    % get the phase locking value
    if length(files)>0
        disp(['Processing Day ' num2str(i) ' CL'])
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
        % run permutation test and get pvalue for each channel
        [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    else
        pac=[];
        rboot=[];
    end

    

    %sum(pac_r>0.3)/253
    pval_cl(i,:) = pval;
    pac_cl(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'CL';
    pac_raw_values(k).Day = i;
    k=k+1;

    % %%%%%% getting batch udpated (CL2) files now
    % folders = session_data(i).folders(batch_idx1);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    % 
    % if ~isempty(files)
    % 
    %     % get the phase locking value
    %     disp(['Processing Day ' num2str(i) ' Batch'])
    %     [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    % 
    %     % run permutation test and get pvalue for each channel
    %     [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    % 
    %     pval_batch(i,:) = pval;
    %     pac_batch(i,:) = abs(mean(pac));
    %     %rboot_batch(i,:,:) = rboot;
    %     pac_raw_values(k).pac = pac;
    %     pac_raw_values(k).boot = rboot;
    %     pac_raw_values(k).type = 'Batch';
    %     pac_raw_values(k).Day = i;
    %     k=k+1;
    % 
    % 
    % else
    %     pac_batch(i,:)=NaN(1,253);
    %     pval_batch(i,:)=NaN(1,253);
    % end

end

toc


%cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
save PAC_B6_LFO_hG_rawValues_New -v7.3


% plotting results
cl_days=[2:2:length(pac_raw_values)];
pac_all=[];
cl=[];
for i=1:length(cl_days)
    tmp = pac_raw_values(cl_days(i)).pac;
    tmp = abs(mean(tmp));
    pac_all(i,:) = tmp;

    ptmp=pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);    
    %pfdr = 0.013;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
end

figure;boxplot(pac_all')
xticks(1:size(pac_all,1))
xlabel('Days')
ylabel('PAC mu hG')
xlim([0.5 11.5])

figure;plot(1:size(pac_all,1),cl)
xticks(1:size(pac_all,1))
xlabel('Days')
ylabel('No. sig chan')
xlim([0.5 11.5])



%% (MAIN) GETTING PAC BETWEEN MU AND HG IN ARROW TASK
% B3

% 
d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3); % 8 to 10 or 0.5 to 5



% d1 = designfilt('lowpassiir', 'FilterOrder', 4, ...
%                'HalfPowerFrequency', 3, 'SampleRate', 1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

pac_ol=[];pval_ol=[];
pac_cl=[];pval_cl=[];
pac_batch=[];pval_batch=[];
rboot_ol=[];rboot_cl=[];rboot_batch=[];
pac_raw_values={};k=1;
len_days = min(11,length(session_data));

for i=1:len_days


    days=i;
    disp(['Processing day ' num2str(days)])

    folders_imag =  strcmp(session_data(days).folder_type,'I');
    folders_online = strcmp(session_data(days).folder_type,'O');
    folders_batch = strcmp(session_data(days).folder_type,'B');
    folders_batch1 = strcmp(session_data(days).folder_type,'B1');
    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    batch_idx1 = find(folders_batch1==1);
    online_idx=[online_idx batch_idx batch_idx1];
    %online_idx=[online_idx batch_idx batch_idx1];
    %online_idx = [batch_idx batch_idx1];



    %%%%%% get imagined data files
    folders = session_data(days).folders(imag_idx);
    day_date = session_data(days).Day;
    files=[];
    for ii=1:length(folders)
        %folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    len = min(120,length(files));
    idx=randperm(length(files),len);
    files=files(idx);

    % get the phase locking value
    if length(files)>0
        disp(['Processing Day ' num2str(i) ' OL'])
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    else
        pac=[];
    end


    % run permutation test and get pvalue for each channel
    %[pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    pval=[];
    rboot=[];

    %sum(pac_r>0.3)/253
    %pval_ol(i,:) = pval;
    %pac_ol(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'OL';
    pac_raw_values(k).Day = i;
    k=k+1;


    %%%%%% get online data files %%%%%
    folders = session_data(days).folders(online_idx);
    day_date = session_data(days).Day;
    files=[];
    for ii=1:length(folders)
        %folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end
  
    len = min(120,length(files));
    idx=randperm(length(files),len);
    files=files(idx);

    % get the phase locking value
    if length(files)>0
        disp(['Processing Day ' num2str(i) ' CL'])
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
        % run permutation test and get pvalue for each channel
        [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    else
        pac=[];
        rboot=[];
    end

    

    %sum(pac_r>0.3)/253
    pval_cl(i,:) = pval;
    pac_cl(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'CL';
    pac_raw_values(k).Day = i;
    k=k+1;

    % %%%%%% getting batch udpated (CL2) files now
    % folders = session_data(i).folders(batch_idx1);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    % 
    % if ~isempty(files)
    % 
    %     % get the phase locking value
    %     disp(['Processing Day ' num2str(i) ' Batch'])
    %     [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    % 
    %     % run permutation test and get pvalue for each channel
    %     [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    % 
    %     pval_batch(i,:) = pval;
    %     pac_batch(i,:) = abs(mean(pac));
    %     %rboot_batch(i,:,:) = rboot;
    %     pac_raw_values(k).pac = pac;
    %     pac_raw_values(k).boot = rboot;
    %     pac_raw_values(k).type = 'Batch';
    %     pac_raw_values(k).Day = i;
    %     k=k+1;
    % 
    % 
    % else
    %     pac_batch(i,:)=NaN(1,253);
    %     pval_batch(i,:)=NaN(1,253);
    % end

end




%cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
save PAC_B3_mu_hG_rawValues_Arrow_New -v7.3


% plotting results
cl_days=[2:2:length(pac_raw_values)];
pac_all=[];
cl=[];
for i=1:length(cl_days)
    tmp = pac_raw_values(cl_days(i)).pac;
    tmp = abs(mean(tmp));
    pac_all(i,:) = tmp;

    ptmp=pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);    
    %pfdr = 0.013;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
end

figure;boxplot(pac_all')
xticks(1:size(pac_all,1))
xlabel('Days')
ylabel('PAC mu hG')
xlim([0.5 11.5])

figure;plot(1:size(pac_all,1),cl)
xticks(1:size(pac_all,1))
xlabel('Days')
ylabel('No. sig chan')
xlim([0.5 11.5])


