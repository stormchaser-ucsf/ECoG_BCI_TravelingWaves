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

    %idx = randperm(length(files),20);
    idx = 1:length(files);
    files_days=[files_days;files(idx)];
end

%%%% running stats
tic
idx=1:length(files_days);
[hold_mu,hold_wave_len] = ...
    planar_waves_stats_onlyWave(files_days(idx),d2,hilbert_flag,ecog_grid,...
    grid_layout,elecmatrix,bpFilt,d1,1);
toc

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

%% EC189

clc;clear

addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))

% loading data
get_data_analyses_EC189
imaging_EC189
close all
cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189/')
ecog_grid=[];
k=1;
for i=1:16:256
    ecog_grid(k,:) = i:i+15;
    k=k+1;
end
ecog_grid = flipud(ecog_grid);
parallel_check


bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',5.5,'HalfPowerFrequency2',8.5, ...
    'SampleRate',Fs);

% bpFilt = designfilt('lowpassiir', 'FilterOrder', 4, ...
%                'HalfPowerFrequency', 3, 'SampleRate', Fs);

hGFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',Fs);



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
stats_move={};
kin_var_waves=[];kin_var_nonwaves=[];
hold_mu={};
hold_wave_len=[];
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
    st = st+0.3;
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
    hold_mu{i} = df;

    % get kinematic data and downsample
    index = (kin_time_resample >= st) .* (kin_time_resample<=stp);
    kindata = kindata_full_length_resampled(logical(index),13:15);
    tx = (1/Fs)*[0:size(kindata,1)-1];
    kindata = resample(kindata,tx,50); %hilbert of mu
    kindata = kindata(:,1);
    %[c,s,l]=pca(kindata);


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
    dcyc_hold =  [dcyc_hold dcyc];
    hold_wave_len = [hold_wave_len sum(out)];

    %%%% get mu power and plv within wave periods
    tmp={};tmp_mu={};I=ones(length(stab1)+1,1);
    tmp_hg_mu={};
    tmp_kin={};
    for k=1:length(st)
        tmp{k} = df_pow(st(k):stp(k),bad_chI); %mu amplitude
        tmp_mu{k} = df(st(k):stp(k),bad_chI); %hilbert of mu
        tmp_hg_mu{k} = hg_mu_data(st(k):stp(k),bad_chI); %hilbert of hg mu
        I(st(k):stp(k))=0;
        %tmp_kin{k} = std(kindata(st(k):stp(k)))/mean(kindata(st(k):stp(k)));
        tmp_kin{k} = (kindata(st(k):stp(k)));
    end
    % power
    tmp = cell2mat(tmp');
    mu_pow_waves = [mu_pow_waves ;nanmean(tmp,1)];

    %plv
    tmp_mu = angle(cell2mat(tmp_mu'));
    tmp_hg_mu = angle(cell2mat(tmp_hg_mu'));
    res_wave = (exp(1i .* (tmp_mu - tmp_hg_mu)));
    kin_waves = std(cell2mat(tmp_kin'));
    stats(i).plv_wave = res_wave;
    stats(i).kin_wave = kin_waves;
    kin_var_waves(i) = kin_waves;


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

    % get kinematic var during non wave periods
    I = [0;diff(I)];
    [aa bb] = find(I==1);
    [aa1 bb1] = find(I==-1);
    tmp_kin={};
    for k=1:length(aa1)
        %tmp_kin{k} = std(kindata(aa(k):aa1(k)))/mean(kindata(aa(k):aa1(k)));
        tmp_kin{k} = (kindata(aa(k):aa1(k)));
    end
    kin_nonwaves = std(cell2mat(tmp_kin'));
    kin_var_nonwaves(i) = kin_nonwaves;
    stats(i).kin_nonwave = kin_nonwaves;




    %%%%% MOVE PERIOD %%%%%%
    st = trial_timings(i).movement.cue(3).time; %anin
    st=st+0.5;
    stp = st+3;
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
    t = length(stab1) * 20/1e3;
    ff1 = length(out)/t; % frequency/s
    d = mean(out) * 20/1e3; %duration in s
    dcyc=ff1*d;
    dcyc_move =  [dcyc_move dcyc];

    %%% new
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
    %mu_pow_waves = [mu_pow_waves ;nanmean(tmp,1)];

    %plv
    tmp_mu = angle(cell2mat(tmp_mu'));
    tmp_hg_mu = angle(cell2mat(tmp_hg_mu'));
    res_wave = (exp(1i .* (tmp_mu - tmp_hg_mu)));
    stats_move(i).plv_wave = res_wave;


    %%%% get mu power within non wave periods
    I = logical(I);
    % power
    tmp_nonwave = df_pow(I,bad_chI);
    %mu_pow_non_waves = [mu_pow_non_waves ;nanmean(tmp_nonwave,1)];
    % plv
    tmp_mu = angle(df(I,bad_chI));
    tmp_hg_mu = angle(hg_mu_data(I,bad_chI));
    res_nonwave = (exp(1i .* (tmp_mu - tmp_hg_mu)));
    stats_move(i).plv_nonwave = res_nonwave;

end

% null statistics for the duration of wave events in the real data
stat = sum(hold_wave_len);
boot=[];
for iter=1:250
    disp(['Processing null iteration ' num2str(iter)])
    %hold_mu_shuff = full_spatiotemporal_shuffle(hold_mu);
    %hold_mu_shuff = temporal_shuffle_complex_mu(hold_mu); %GOAT
    hold_mu_shuff = temporal_shuffle_all_trials_no_replace(hold_mu);
    %hold_mu_shuff = spatiotemporal_shuffle_complex_mu(hold_mu);
    hold_wave_len_tmp=[];
    parfor j=1:length(hold_mu_shuff)
        df = hold_mu_shuff{j};
        % detect waves based on spatiotemporally stable phase gradients
        planar_val_time=[];
        for t=1:size(df,1)
            % estimate planar waves across grid
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
        hold_wave_len_tmp(j) = sum(out);
    end
    boot(iter) = sum(hold_wave_len_tmp);
end
max(1/length(boot),(sum(stat<=boot))/length(boot))

figure;hist(boot*20/1e3)
vline(stat*20/1e3)

save wave_duration_stats_EC189 stat boot -v7.3


%% EC176


clc;clear
close all
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))
% loading data
get_data_analyses_EC176
imaging_EC176
close all
cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC176_ProcessingForNikhilesh/ecog_data_NN')

lfp=lfp(10e5:end,:);
lfp_time=lfp_time(10e5:end);

ecog_grid=[];
k=1;
for i=1:16:256
    ecog_grid(k,:) = i:i+15;
    k=k+1;
end
ecog_grid = flipud(ecog_grid);
parallel_check

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',14.5,'HalfPowerFrequency2',17.5, ...
    'SampleRate',Fs);
% bpFilt = designfilt('lowpassiir', 'FilterOrder', 4, ...
%                'HalfPowerFrequency', 3, 'SampleRate', Fs);
% bpFilt = designfilt('lowpassiir', 'FilterOrder', 4, ...
%                'HalfPowerFrequency', 3, 'SampleRate', Fs);
hGFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',Fs);

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

dcyc_move = [];
dcyc_hold = [];
wave_f = [];
wave_dur = [];
mu_pow_waves=[];
mu_pow_non_waves=[];
stats={};
hold_wave_len=[];
hold_mu={};
for i=1:length(trial_timings)
    timings = [trial_timings(i).movement.cue.time];
    % for j=1:length(timings)
    %     vline(timings(1),'r') % rest
    %     vline(timings(2),'y') % hold
    %     vline(timings(3),'g') % go
    % end


    %%%% HOLD PERIOD %%%%%%
    st = trial_timings(i).movement.cue(2).time; %anin
    st = st+0.5;
    %stp=st+2.5;
    stp = trial_timings(i).movement.cue(3).time; %anin
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
    hold_mu{i} = df;

    % detect waves based on spatiotemporally stable phase gradients
    planar_val_time=[];
    parfor t=1:size(df,1)
        % estimate planar waves across mini grid
        tmp = df(t,:);
        %tmp(bad_ch)= NaN + 1i*NaN;
        xph = tmp(ecog_grid);
        %[planar_val,aa,bb] = planar_stats_muller_intact(xph,mask);
        [planar_val,aa,bb] = planar_stats_muller(xph);
        %planar_val(mask==0) = NaN +1i*NaN;
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
    dcyc_hold =  [dcyc_hold dcyc];
    wave_f(i) = ff1;
    wave_dur(i) = d;
    hold_wave_len = [hold_wave_len; sum(out)];

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
        %tmp(bad_ch)= NaN + 1i*NaN;
        xph = tmp(ecog_grid);
        %[planar_val,aa,bb] = planar_stats_muller_intact(xph,mask);
        [planar_val,aa,bb] = planar_stats_muller(xph);
        %planar_val(mask==0) = NaN +1i*NaN;
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

% null statistics for the duration of wave events in the real data
stat = sum(hold_wave_len);
boot=[];
for iter=1:250
    disp(['Processing null iteration ' num2str(iter)])
    %hold_mu_shuff = full_spatiotemporal_shuffle(hold_mu);
    %hold_mu_shuff = temporal_shuffle_complex_mu(hold_mu); %GOAT
    hold_mu_shuff = temporal_shuffle_all_trials_no_replace(hold_mu);
    %hold_mu_shuff = spatiotemporal_shuffle_complex_mu(hold_mu);
    hold_wave_len_tmp=[];
    parfor j=1:length(hold_mu_shuff)
        df = hold_mu_shuff{j};
        % detect waves based on spatiotemporally stable phase gradients
        planar_val_time=[];
        for t=1:size(df,1)
            tmp = df(t,:);
            %tmp(bad_ch)= NaN + 1i*NaN;
            xph = tmp(ecog_grid);
            %[planar_val,aa,bb] = planar_stats_muller_intact(xph,mask);
            [planar_val,aa,bb] = planar_stats_muller(xph);
            %planar_val(mask==0) = NaN +1i*NaN;
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
        hold_wave_len_tmp(j) = sum(out);
    end
    boot(iter) = sum(hold_wave_len_tmp);
end
max(1/length(boot),(sum(stat<=boot))/length(boot))

figure;hist(boot*20/1e3)
vline(stat*20/1e3)

save wave_duration_stats_EC176 stat boot -v7.3

%% EC210

clc;clear
close all
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))


% loading data
get_data_analyses_EC210
imaging_EC210
close all
cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210')
ecog_grid=[];
k=1;
for i=1:16:256
    ecog_grid(k,:) = i:i+15;
    k=k+1;
end
ecog_grid = flipud(fliplr(ecog_grid'));
lfp=zscore(lfp);

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',4.5,'HalfPowerFrequency2',7.5, ...
    'SampleRate',Fs);
% bpFilt = designfilt('lowpassiir', 'FilterOrder', 4, ...
%                'HalfPowerFrequency', 3, 'SampleRate', Fs);
hGFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',Fs);

ch=1:256;
%ch=129:256;
%bad_ch_orig = bad_ch;
%ecog_grid_orig = ecog_grid;
%bad_ch = bad_ch(bad_ch>128);
%ecog_grid=ecog_grid(:,1:8);
%bad_chI=bad_chI(129:end);

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

dcyc_move = [];
dcyc_hold = [];
wave_f = [];
wave_dur = [];
mu_pow_waves=[];
mu_pow_non_waves=[];
hold_mu={};
hold_wave_len=[];
stats={};
for i=1:length(trial_timings)
    timings = [trial_timings(i).movement.cue.time];
    % for j=1:length(timings)
    %     vline(timings(1),'r') % rest
    %     vline(timings(2),'y') % hold
    %     vline(timings(3),'g') % go
    % end


    %%%% HOLD PERIOD %%%%%%
    st = trial_timings(i).movement.cue(2).time; %anin
    %st=st-0.5;
    stp=st+3.0;
    %st = st-0.5;
    %stp = trial_timings(i).movement.cue(2).time; %anin
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
    hold_mu{i} = df;

    % detect waves based on spatiotemporally stable phase gradients
    planar_val_time=[];
    parfor t=1:size(df,1)
        % estimate planar waves across mini grid
        tmp = df(t,:);
        %tmp(bad_ch)= NaN + 1i*NaN;
        xph = tmp(ecog_grid);
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
    dcyc_hold =  [dcyc_hold dcyc];
    wave_f(i) = ff1;
    wave_dur(i) = d;
    hold_wave_len = [hold_wave_len sum(out)];

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
        %tmp(bad_ch)= NaN + 1i*NaN;
        xph = tmp(ecog_grid);
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


% null statistics for the duration of wave events in the real data
stat = sum(hold_wave_len);
boot=[];
for iter=1:250
    disp(['Processing null iteration ' num2str(iter)])
    %hold_mu_shuff = full_spatiotemporal_shuffle(hold_mu);
    %hold_mu_shuff = temporal_shuffle_complex_mu(hold_mu); %GOAT
    hold_mu_shuff = temporal_shuffle_all_trials_no_replace(hold_mu);
    %hold_mu_shuff = spatiotemporal_shuffle_complex_mu(hold_mu);
    hold_wave_len_tmp=[];
    parfor j=1:length(hold_mu_shuff)
        df = hold_mu_shuff{j};
        % detect waves based on spatiotemporally stable phase gradients
        planar_val_time=[];
        for t=1:size(df,1)
            % estimate planar waves across grid
            tmp = df(t,:);
            %tmp(bad_ch)= NaN + 1i*NaN;
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
        hold_wave_len_tmp(j) = sum(out);
    end
    boot(iter) = sum(hold_wave_len_tmp);
end

max(1/length(boot),(sum(stat<=boot))/length(boot))

figure;
hist(boot*20/1e3)
vline(stat*20/1e3)

save wave_duration_stats_EC210 stat boot -v7.3


%% PLOTTING THE EC SUBJECTS

clear
clc
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves'))
cd('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves')
addpath('C:\Users\nikic\Documents\MATLAB')
if ispc
    a=load('wave_duration_stats_EC210.mat');
    b=load('wave_duration_stats_EC176.mat');
    c=load('wave_duration_stats_EC189.mat');
else
    a=load('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210/wave_duration_stats_EC210.mat');
    b=load('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC176_ProcessingForNikhilesh/ecog_data_NN/wave_duration_stats_EC176.mat');
    c=load('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189/wave_duration_stats_EC189.mat');
end

figure;
hold on
hist(a.boot*20/1e3)
vline(a.stat*20/1e3)
xlabel('Time (s)')
ylabel('Count')
title('EC 210')
xlim([10 38])
xticks(0:5:40)
plot_beautify
figure
hist(b.boot*20/1e3)
vline(b.stat*20/1e3)
xlabel('Time (s)')
ylabel('Count')
title('EC 176')
xlim([10 38])
xticks(0:5:40)
plot_beautify
figure
hist(c.boot*20/1e3)
vline(c.stat*20/1e3)
xlabel('Time (s)')
ylabel('Count')
title('EC 189')
xlim([10 38])
xticks(0:5:40)
plot_beautify

%% SAVING TO DISK


cd('C:\Users\nikic\Documents\Ganguly lab\ECoG BCI\BCI_Paper_Waves_Hand\Paper_text\Figures_New_v2\Figure3\')

% svg
set(gcf,'PaperPositionMode','auto');
print(gcf,'PMd_Channels_grid.svg','-dsvg','-painters','-r300');

% png
set(gcf,'PaperPositionMode','auto');
print(gcf,'STG_Channels.png','-dpng','-r1200');
