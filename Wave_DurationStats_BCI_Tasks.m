% wave duration stats in BCI tasks

%% B3 hand

clc;clear
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
cd(root_path)
load session_data_B3_Hand
load('ECOG_Grid_8596_000067_B3.mat')
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);
d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',50);
deltaFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',50);
bpFilt=deltaFilt;
hilbert_flag=1;
imaging_B3_waves;
len_days = min(11,length(session_data));
num_targets=12;
close all


parallel_check
files_days=[];
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

    idx = randperm(length(files),20);
    files_days=[files_days;files(idx)];
end

%%%% running stats
idx=1:length(files_days);
[hold_mu,hold_wave_len] = ...
    planar_waves_stats_onlyWave(files_days(idx),d2,hilbert_flag,ecog_grid,...
    grid_layout,elecmatrix,bpFilt,d1,1);

% null statistics for the duration of wave events in the real data
stat = sum(hold_wave_len);
boot=[];
for iter=3:100
    disp(['Processing null iteration ' num2str(iter)])
    %hold_mu_shuff = temporal_shuffle_complex_mu(hold_mu); %GOAT
    hold_mu_shuff = temporal_shuffle_all_trials_no_replace(hold_mu);%GOAT
    hold_wave_len_tmp=[];
    parfor j=1:length(hold_mu_shuff)
        df = hold_mu_shuff{j};
        % detect waves based on spatiotemporally stable phase gradients
        planar_val_time=[];
        for t=1:size(df,1)
            tmp = df(t,:);
            xph = tmp(ecog_grid);
            [planar_val,aa,bb] = planar_stats_muller(xph);
            planar_val_time(t,:,:) = planar_val;
        end
        stab=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));
        end
        % stab1 = zscore(stab(1:end));
        % [out,st,stp] = wave_stability_detect(stab1,0,3);
        stab = zscore(stab(100:end)); % 100:end for B3 HAND
        [out,st,stp] = wave_stability_detect(stab,0,3);
        st=st+99;%for b3 hand
        stp=stp+99;%for b3 hand
        hold_wave_len_tmp(j) = sum(out);
    end
    boot(iter) = sum(hold_wave_len_tmp);
end
max(1/length(boot),(sum(stat<=boot))/length(boot))

figure;hist(boot*20/1e3)
vline(stat*20/1e3)

save wave_duration_stats_B3_Hand stat boot -v7.3


%% B3 Arrow


clc;clear
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
cd(root_path)
load session_data_B3
load('ECOG_Grid_8596_000067_B3.mat')
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);
d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',50);
deltaFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',50);
bpFilt=deltaFilt;
hilbert_flag=1;
imaging_B3_waves;
len_days = min(11,length(session_data));
num_targets=7;
close all
parallel_check


files_days=[];
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

    idx = randperm(length(files),20);
    files_days=[files_days;files(idx)];
end

%%%% running stats
idx=1:length(files_days);
[hold_mu,hold_wave_len] = ...
    planar_waves_stats_onlyWave(files_days(idx),d2,hilbert_flag,ecog_grid,...
    grid_layout,elecmatrix,bpFilt,d1,1);

% null statistics for the duration of wave events in the real data
stat = sum(hold_wave_len);
boot=[];
for iter=1:100
    disp(['Processing null iteration ' num2str(iter)])
    %hold_mu_shuff = temporal_shuffle_complex_mu(hold_mu); %GOAT
    hold_mu_shuff = temporal_shuffle_all_trials_no_replace(hold_mu);%GOAT
    hold_wave_len_tmp=[];
    parfor j=1:length(hold_mu_shuff)
        df = hold_mu_shuff{j};
        % detect waves based on spatiotemporally stable phase gradients
        planar_val_time=[];
        for t=1:size(df,1)
            tmp = df(t,:);
            xph = tmp(ecog_grid);
            [planar_val,aa,bb] = planar_stats_muller(xph);
            planar_val_time(t,:,:) = planar_val;
        end
        stab=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));
        end
        % stab1 = zscore(stab(1:end));
        % [out,st,stp] = wave_stability_detect(stab1,0,3);
        % stab = zscore(stab(100:end)); % 100:end for B3 HAND
        % [out,st,stp] = wave_stability_detect(stab,0,3);
        % st=st+99;%for b3 hand
        % stp=stp+99;%for b3 hand

        stab1 = zscore(stab(15:end));
        [out,st,stp] = wave_stability_detect(stab1,0,3);
        st = st+14;
        stp = stp+14;
        hold_wave_len_tmp(j) = sum(out);
    end
    boot(iter) = sum(hold_wave_len_tmp);
end
max(1/length(boot),(sum(stat<=boot))/length(boot))

figure;hist(boot*20/1e3)
vline(stat*20/1e3)

save wave_duration_stats_B3_Arrow stat boot -v7.3

%% B1 arrow

clc;clear
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
cd(root_path)
load('ECOG_Grid_8596_000067_B3.mat')
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);
d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',50);
deltaFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',50);
bpFilt=deltaFilt;
hilbert_flag=1;
imaging_B3_waves;

folders={'20240515', '20240517', '20240614', ...
    '20240619', '20240621', '20240626',...
    '20240710','20240712','20240731'};
folders=folders(1:end-1);
num_targets=7;
close all
parallel_check

files_days=[];
for days=1:length(folders)

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
    
    %%%%%% get online data files %%%%%
    files=[];
    for ii=1:length(online_idx)
        online_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',online_folderpath)'];
    end
   
    idx = randperm(length(files),25);
    files_days=[files_days;files(idx)];
end

%%%% running stats
idx=1:length(files_days);
[hold_mu,hold_wave_len] = ...
    planar_waves_stats_onlyWave(files_days(idx),d2,hilbert_flag,ecog_grid,...
    grid_layout,elecmatrix,bpFilt,d1,1);

% null statistics for the duration of wave events in the real data
stat = sum(hold_wave_len);
boot=[];
for iter=1:100
    disp(['Processing null iteration ' num2str(iter)])
    %hold_mu_shuff = temporal_shuffle_complex_mu(hold_mu); %GOAT
    hold_mu_shuff = temporal_shuffle_all_trials_no_replace(hold_mu);%GOAT
    hold_wave_len_tmp=[];
    parfor j=1:length(hold_mu_shuff)
        df = hold_mu_shuff{j};
        % detect waves based on spatiotemporally stable phase gradients
        planar_val_time=[];
        for t=1:size(df,1)
            tmp = df(t,:);
            xph = tmp(ecog_grid);
            [planar_val,aa,bb] = planar_stats_muller(xph);
            planar_val_time(t,:,:) = planar_val;
        end
        stab=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));
        end
        % stab1 = zscore(stab(1:end));
        % [out,st,stp] = wave_stability_detect(stab1,0,3);
        % stab = zscore(stab(100:end)); % 100:end for B3 HAND
        % [out,st,stp] = wave_stability_detect(stab,0,3);
        % st=st+99;%for b3 hand
        % stp=stp+99;%for b3 hand

        stab1 = zscore(stab(15:end));
        [out,st,stp] = wave_stability_detect(stab1,0,3);
        st = st+14;
        stp = stp+14;
        hold_wave_len_tmp(j) = sum(out);
    end
    boot(iter) = sum(hold_wave_len_tmp);
end
max(1/length(boot),(sum(stat<=boot))/length(boot))

figure;hist(boot*20/1e3)
vline(stat*20/1e3)

save wave_duration_stats_B1_Arrow stat boot -v7.3



%% B6 arrow

clc;clear
root_path='/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
cd(root_path)
load('ECOG_Grid_8596_000067_B3.mat')
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);
d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',50);
deltaFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',50);
bpFilt=deltaFilt;
hilbert_flag=1;
imaging_B3_waves;

  folders = {'20250530','20250610','20250624','20250703','20250708','20250717',...
        '20250917','20250924','20251203','20251204','20251210','20260116'};
  


num_targets=7;
close all
parallel_check

files_days=[];
for days=1:length(folders)

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
    
    %%%%%% get online data files %%%%%
    files=[];
    for ii=1:length(online_idx)
        online_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',online_folderpath)'];
    end
   
    idx = randperm(length(files),25);
    files_days=[files_days;files(idx)];
end

%%%% running stats
idx=1:length(files_days);
[hold_mu,hold_wave_len] = ...
    planar_waves_stats_onlyWave(files_days(idx),d2,hilbert_flag,ecog_grid,...
    grid_layout,elecmatrix,bpFilt,d1,1);

% null statistics for the duration of wave events in the real data
stat = sum(hold_wave_len);
boot=[];
for iter=1:100
    disp(['Processing null iteration ' num2str(iter)])
    %hold_mu_shuff = temporal_shuffle_complex_mu(hold_mu); %GOAT
    hold_mu_shuff = temporal_shuffle_all_trials_no_replace(hold_mu);%GOAT
    hold_wave_len_tmp=[];
    parfor j=1:length(hold_mu_shuff)
        df = hold_mu_shuff{j};
        % detect waves based on spatiotemporally stable phase gradients
        planar_val_time=[];
        for t=1:size(df,1)
            tmp = df(t,:);
            xph = tmp(ecog_grid);
            [planar_val,aa,bb] = planar_stats_muller(xph);
            planar_val_time(t,:,:) = planar_val;
        end
        stab=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));
        end
        % stab1 = zscore(stab(1:end));
        % [out,st,stp] = wave_stability_detect(stab1,0,3);
        % stab = zscore(stab(100:end)); % 100:end for B3 HAND
        % [out,st,stp] = wave_stability_detect(stab,0,3);
        % st=st+99;%for b3 hand
        % stp=stp+99;%for b3 hand

        stab1 = zscore(stab(15:end));
        [out,st,stp] = wave_stability_detect(stab1,0,3);
        st = st+14;
        stp = stp+14;
        hold_wave_len_tmp(j) = sum(out);
    end
    boot(iter) = sum(hold_wave_len_tmp);
end
max(1/length(boot),(sum(stat<=boot))/length(boot))

figure;hist(boot*20/1e3)
vline(stat*20/1e3)

save wave_duration_stats_B6_Arrow stat boot -v7.3

