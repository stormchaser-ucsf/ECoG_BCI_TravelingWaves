%% FOR FIG 2
% goal here is to highlight ERPs and 1/f for hG and also for mu, in BCI and
% intact individuals. While mu has a power increase during imagined BCI
% movements, there is ERD in intact movements.

%% INIT
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

%% LOOK AT SMOOTHED NEURAL FEATURES ALPHA BAND ACTIVITY
% arrow / hand task

root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker';
subfolders = '20240712/Robot3DArrow';
filepath=fullfile(root_path,subfolders);
cd(filepath)

load('ECOG_Grid_8596_000067_B3.mat')

files = findfiles('.mat',filepath,1)';

files1=[];
for i=1:length(files)
    if isempty(regexp(files{i},'kf_params'))
        files1=[files1;files(i)];
    end
end
files=files1;


% generate ERPs
load(files{1})
Params = TrialData.Params;
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);
ERP=[];
ERP_beta=[];
ERP_hg=[];

ERP_beta_NF=[];
ERP_hG_NF=[];
for ii=127:length(files)
    disp(ii/length(files)*100)
    loaded=true;
    try
        load(files{ii})
    catch
        loaded=false;
    end

    if loaded && (TrialData.TargetID==1 )

        data_trial = (TrialData.BroadbandData');
        task_state = TrialData.TaskState;
        kinax1 = find(task_state==1);
        kinax2 = find(task_state==2);
        kinax3 = find(task_state==3);
        kinax4 = find(task_state==4);

        data0 = cell2mat(data_trial(kinax1));
        data1 = cell2mat(data_trial(kinax2));
        data2 = cell2mat(data_trial(kinax3));
        data3 = cell2mat(data_trial(kinax4));
        l0 = size(data0,1);
        l1 = size(data1,1);
        l2 = size(data2,1);
        l3 = size(data3,1);

        %neural features
        tmp_data = TrialData.SmoothedNeuralFeatures;
        tmp_data = tmp_data(1:16);
        tmp_data = cell2mat(tmp_data)';

        tmp_data_alpha = tmp_data(:,769:1024);%beta 1025:1280, alpha 769:1024
        m = mean(tmp_data_alpha(1:5,:),1);
        s = std(tmp_data_alpha(1:5,:),1);
        tmp_data_alpha = (tmp_data_alpha-m)./s;
        ERP_beta_NF= cat(3,ERP_beta_NF,tmp_data_alpha);


        tmp_data_hg = tmp_data(:,1537:end);
        m = mean(tmp_data_hg(1:5,:),1);
        s = std(tmp_data_hg(1:5,:),1);
        tmp_data_hg = (tmp_data_hg-m)./s;
        ERP_hG_NF= cat(3,ERP_hG_NF,tmp_data_hg);



        data_trial = cell2mat(data_trial);
        data_trial0=data_trial;

        % filter in mu
        data_trial = filter(bpFilt,data_trial);
        data_trial = abs(hilbert(data_trial));

        % filter in beta
        tmp_beta=[];
        for j=4:5
            tmp = filtfilt(Params.FilterBank(j).b,Params.FilterBank(j).a,...
                data_trial0);
            tmp = abs(hilbert(tmp));
            tmp_beta = cat(3,tmp_beta,tmp);
        end
        tmp_beta = squeeze(mean(tmp_beta,3));

        % filter in hG
        tmp_hg=[];
        for j=9:16
            tmp = filtfilt(Params.FilterBank(j).b,Params.FilterBank(j).a,...
                data_trial0);
            tmp = abs(hilbert(tmp));
            tmp_hg = cat(3,tmp_hg,tmp);
        end
        tmp_hg = squeeze(mean(tmp_hg,3));

        % segment and epoch
        l0=200; % removing the first 200ms
        data_ep = data_trial(l0+1:end,:); % take from state 1 onwards
        data_ep_beta = tmp_beta(l0+1:end,:); % take from state 1 onwards
        data_ep_hg = tmp_hg(l0+1:end,:); % take from state 1 onwards

        % zscore
        m = mean(data_ep(1:800,:),1);
        s = std(data_ep(1:800,:),1);
        data_ep = (data_ep-m)./s;

        m = mean(data_ep_beta(1:800,:),1);
        s = std(data_ep_beta(1:800,:),1);
        data_ep_beta = (data_ep_beta-m)./s;

        m = mean(data_ep_hg(1:800,:),1);
        s = std(data_ep_hg(1:800,:),1);
        data_ep_hg = (data_ep_hg-m)./s;

        % store
        ERP = cat(3,ERP,data_ep(1:3.2e3,:));
        ERP_beta = cat(3,ERP_beta,data_ep_beta(1:3.2e3,:));
        ERP_hg = cat(3,ERP_hg,data_ep_hg(1:3.2e3,:));

    end

end

% arrow imagined
% first 800 is state 1
% next 1000 is state 2
% next 5000 is state 3
% next 500 is state 4

tmp = squeeze(mean(ERP,3));
tmp_beta = squeeze(mean(ERP_beta,3));
tmp_hg = squeeze(mean(ERP_hg,3));



ch=[137	143	148	152
    159	160	30	28
    134	140	170	174
    49	45	41	38
    221	62	59	56];
%ch=137;
% ch=[185	188	191	64	61
% 165	168	172	176	180
% 43	39	36	33	161
% 94	90	86	83	80
% 210	213	216	220	224];

figure;hold on
%plot(tmp(:,ch),'LineWidth',1)
%plot(tmp_beta(:,ch),'LineWidth',1)
plot(mean(tmp(:,ch),2),'LineWidth',2)
plot(mean(tmp_beta(:,ch),2),'LineWidth',2)
plot(mean(tmp_hg(:,ch),2),'LineWidth',2)
%vline([1e3 2e3])
vline([800 1800 ])
plot_beautify
%xlim([0 4000])
hline(0,'--r')
legend({'Mu','Alpha','hG'})
xlabel('Time (ms)')
ylabel('Amplit.')
title('Avg all M1/S1 channels')

figure;
hold on
ch=140;
tmp_beta = squeeze(mean(ERP_beta_NF,3));
tmp_hg = squeeze(mean(ERP_hG_NF,3));
plot(mean(tmp_beta(:,ch),2),'LineWidth',2)
plot(mean(tmp_hg(:,ch),2),'LineWidth',2)
%vline([1e3 2e3])
vline([5 10])
plot_beautify
%xlim([0 4000])
hline(0,'--r')
legend({'Alpha','hG'})
xlabel('Time (ms)')
ylabel('Amplit.')
title('Avg all M1/S1 channels')




%% imaging;
imaging_B3_waves;
close all


%% LOOK AT 1/F AND POWER IN STATE 3 VS. 1
%BCI - B3

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',13,'HalfPowerFrequency2',30, ...
    'SampleRate',1e3); % center freq is 8.5
subj='B3';


if strcmp(subj,'B3')
    % b3
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
    cd(root_path)
    %load session_data_B3_Hand
    load session_data_B3
    load('ECOG_Grid_8596_000067_B3.mat')

    len_days = min(11,length(session_data));
    num_targets=7;
end

if strcmp(subj,'B1')
    % b1
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
    cd(root_path)
    load('ECOG_Grid_8596_000067_B3.mat')
    folders={'20240515', '20240517', '20240614', ...
        '20240619', '20240621', '20240626',...
        '20240710','20240712','20240731'};
    num_targets=7;
end

if strcmp(subj,'B6')
    root_path='/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
    cd(root_path)
    load('ECOG_Grid_8596_000067_B3.mat')
    folders = {'20250530','20250610','20250624','20250703','20250708','20250717',...
        '20250917','20250924','20251203','20251204','20251210','20260116'};
end

pow_days=[];

for days=4%:len_days

    disp(['Processing day ' num2str(days)])


    if strcmp(subj,'B3')

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

    else
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
    end




    % %%%%%% get imagined data files
    % folders = session_data(days).folders(imag_idx);
    % day_date = session_data(days).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
    %     %folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end

    % online data files
    if strcmp(subj,'B3')
        folders = session_data(days).folders(online_idx);
        day_date = session_data(days).Day;
        files=[];
        for ii=1:length(folders)
            %folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
            folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
            %cd(folderpath)
            files = [files;findfiles('mat',folderpath)'];
        end

    else
        files=[];
        for ii=1:length(online_idx)
            imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
            files = [files;findfiles('mat',imag_folderpath)'];
        end
    end

    % perform 1/f over all channels in state 1 and state 3 as comparison
    % also get power in each state
    pow_s1=[];pow_s2=[];pow_s3=[];pow_s4=[];
    for ii=1:length(files)
        disp(ii/length(files)*100)
        loaded=true;
        try
            load(files{ii})
        catch
            loaded=false;
        end

        if loaded% && (TrialData.TargetID==1 )

            data_trial = (TrialData.BroadbandData');
            task_state = TrialData.TaskState;
            kinax1 = find(task_state==1);
            kinax2 = find(task_state==2);
            kinax3 = find(task_state==3);
            kinax4 = find(task_state==4);

            data0 = cell2mat(data_trial(kinax1));
            data1 = cell2mat(data_trial(kinax2));
            data2 = cell2mat(data_trial(kinax3));
            data3 = cell2mat(data_trial(kinax4));
            l0 = size(data0,1);
            l1 = size(data1,1);
            l2 = size(data2,1);
            l3 = size(data3,1);
            l = [l0 l1 l2 l3];
            l = cumsum(l);

            data_trial = cell2mat(data_trial);
            data_trial0=data_trial;

            % filter in mu
            data_trial = filter(d1,data_trial);
            data_trial = abs(hilbert(data_trial));

            % baseline it all to state 1
            m = mean(data_trial(1:l0,:),1);
            s = std(data_trial(1:l0,:),1);
            data_trial = (data_trial-m)./s;

            % get average power within each state
            a = mean(data_trial(1:l(1),:),1);
            pow_s1 = cat(2,pow_s1,a');
            
            a = mean(data_trial(l(1)+1:l(2),:),1);
            pow_s2 = cat(2,pow_s2,a');

            a = mean(data_trial(l(2)+1:l(3),:),1);
            pow_s3 = cat(2,pow_s3,a');

            a = mean(data_trial(l(3)+1:l(4),:),1);
            pow_s4 = cat(2,pow_s4,a');
        end
    end


    a=mean(pow_s1,2);
    b=mean(pow_s2,2);
    c=mean(pow_s3,2);
    d=mean(pow_s4,2);
    figure;
    boxplot([a b c d])
    title(['Day ' num2str(days)])
    plot_beautify
    hline(0,'--r')

end



%% LOOK AT 1/F AND POWER IN STATE 3 VS. 1
%ACTUAL MOTOR CONTROL
% EC189

clc;clear


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

%%% OTHER ANALYSES
% get power of LoMU in move, hold and rest states?
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
    'HalfPowerFrequency1',6,'HalfPowerFrequency2',9, ...
    'SampleRate',Fs);

bpFilt1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',13, ...
    'SampleRate',Fs);


%ch= [204:209 219:224];
ch=1:256;

% get mu and hg
mu = [];hg=[];
alp=[];
mu = filtfilt(bpFilt,lfp(:,ch));
mu = abs(hilbert(mu));
%mu = mean(mu,2);
%mu=zscore(mu);

alp = filtfilt(bpFilt1,lfp(:,ch));
alp = abs(hilbert(alp));
%alp = mean(alp,2);
%alp=zscore(alp);

tmp_hg=[];
for j=1:8
    tmp = filtfilt(filterbank(j).b,filterbank(j).a,...
        lfp(:,ch));
    tmp = abs(hilbert(tmp));
    tmp_hg = cat(3,tmp_hg,tmp);
end
tmp_hg = squeeze(mean(tmp_hg,3));
hg=tmp_hg;
%hg = squeeze(mean(tmp_hg,2));
%hg=zscore(hg);


% get a sense of the timing of events
trial_times=[];
for i=1:length(trial_timings)
    timings = [trial_timings(i).movement.cue.time];
    trial_times(i,:) = timings;
end

trial_times(2:end,1) - trial_times(1:end-1,3);

% plot
% also extract power spectrum within each signal portion (1/f)
figure;
hold on
plot(lfp_time,smooth(zscore(hg(:,250)),100))
plot(lfp_time,smooth(zscore(mu(:,250)),100))
legend({'hG','mu'})

mu_ep=[];
hg_ep=[];
alp_ep=[];
kin_ep=[];
osc_clus=[];
stats=[];
pow_freq=[];
ffreq=[];
hold_dur = [];
lfp_epochs={};
for i=1:length(trial_timings)
    timings = [trial_timings(i).movement.cue.time];
    for j=1:length(timings)
        vline(timings(1),'r') % rest
        vline(timings(2),'y') % hold
        vline(timings(3),'g') % go
    end
    go_time = trial_timings(i).movement.cue(2).time; %anin
    st = go_time-1; % go back 1s into the rest period
    stp = go_time+7; % average hold time is 4s, go period is 3s.

    hold_dur(i) = trial_timings(i).movement.cue(3).time - ...
        trial_timings(i).movement.cue(2).time;

    index = (kin_time_resample >= st) .* (kin_time_resample<=stp);
    kin_ep(:,i) = kindata_full_length_resampled(logical(index),13);

    index = (lfp_time >= st) .* (lfp_time<=stp);
    mu_ep(:,:,i) = mu(logical(index),:);
    hg_ep(:,:,i) = hg(logical(index),:);
    alp_ep(:,:,i) = alp(logical(index),:);

    % only during the go period or hold period for 1/f
    st = trial_timings(i).movement.cue(3).time; %anin
    %stp = trial_timings(i).movement.cue(2).time; %anin
    stp=st+3; % duration
    index = (lfp_time >= st) .* (lfp_time<=stp);
    data = lfp(logical(index),:);

    lfp_epochs{i}=data;

    spectral_peaks=[];
    stats_tmp=[];
    parfor ii=1:size(data,2)
        x = data(:,ii);

        [Pxx,F] = pwelch(x,512,256,512,Fs);
        %[Pxx,F] = pwelch(x,1024,512,1024,1e3);
        %[Pxx,F] = pwelch(x,hamming(1000),500,2048,1000);
        pow_freq = [pow_freq;Pxx' ];
        ffreq = [ffreq ;F'];
        idx = logical((F>0) .* (F<=40));
        F1=F(idx);
        F1=log2(F1);
        power_spect = Pxx(idx);
        power_spect = log2(power_spect);
        tb=fitlm(F1,power_spect,'RobustOpts','on');
        stats_tmp = [stats_tmp tb.Coefficients.pValue(2)];
        bhat = tb.Coefficients.Estimate;
        x = [ones(length(F1),1) F1];
        yhat = x*bhat;

        %plot
        % figure;
        % plot(F1,power_spect,'LineWidth',1);
        % hold on
        % plot(F1,yhat,'LineWidth',1);

        % get peaks in power spectrum at specific frequencies
        power_spect = zscore(power_spect - yhat);
        [aa ,bb]=findpeaks(power_spect);
        peak_loc = bb(find(power_spect(bb)>1));
        freqs = 2.^F1(peak_loc);
        pow = power_spect(peak_loc);

        %store
        spectral_peaks(ii).freqs = freqs;
        spectral_peaks(ii).pow = pow;

    end


    % getting oscillation clusters
    if ~isempty(spectral_peaks)
        osc_clus_tmp=[];
        for f=2:40
            ff = [f-1 f+1];
            tmp=0;ch_tmp=[];
            for j=1:length(spectral_peaks)
                freqs = spectral_peaks(j).freqs;
                for k=1:length(freqs)
                    if ff(1) <= freqs(k)  && freqs(k) <= ff(2)
                        tmp=tmp+1;
                        ch_tmp = [ch_tmp j];
                    end
                end
            end
            osc_clus_tmp = [osc_clus_tmp tmp];
        end
        osc_clus = [osc_clus;osc_clus_tmp];
    end

end

%save lfp_epochs_moveState lfp_epochs Fs bad_chI -v7.3

% plot oscillation clusters
f=2:40;
figure;
hold on
plot(f,osc_clus/227,'Color',[.5 .5 .5 .5],'LineWidth',.5)
plot(f,mean(osc_clus,1)/227,'b','LineWidth',2)
xlabel('Freq.')
ylabel('Prop of channels')
plot_beautify
title('1/f sig.')

% plot power spectrum
figure;
hold on
plot(ffreq(1,:),log10(pow_freq'),'Color',[.5 .5 .5 .5])
plot(ffreq(1,:),log10(mean(pow_freq,1)),'k','LineWidth',2)
xlim([0 200])
ylim([-17 -9])

m = mean(mu_ep(1:254*2,:,:),1);
s = std(mu_ep(1:254*2,:,:),1);
mu_ep = (mu_ep-m)./s;

m = mean(hg_ep(1:254*2,:,:),1);
s = std(hg_ep(1:254*2,:,:),1);
hg_ep = (hg_ep-m)./s;

m = mean(alp_ep(1:254*2,:,:),1);
s = std(alp_ep(1:254*2,:,:),1);
alp_ep = (alp_ep-m)./s;


% getting average hG power over grid 
dur = diff(trial_times')';
hold_dur1 = round(Fs*hold_dur');% in samples, duration of hold
hg_pow=[];
pow_s1=[];
pow_s2=[];
pow_s3=[];
aa=[];len=[];
for i=1:length(hold_dur1)
    %tmp = squeeze(mu_ep(:,:,i)); 
    tmp = squeeze(hg_ep(:,:,i)); 
    % first 1s or 508 samples is rest (baseline)
    %idx = 509: (509+hold_dur1(i));
    idx = (509+hold_dur1(i)):size(tmp,1);
    %hg_pow(:,i) = mean(tmp(idx,:),1);
    hg_pow(:,i) = mean(tmp(idx,:));
    aa=[aa;tmp(idx,:)];
    len=[len;length(idx)];

    
    % states 1,2,3, mu
    tmp = squeeze(mu_ep(:,:,i)); 
    idx=1:508;
    pow_s1(:,i) = mean(tmp(idx,:),1);
    
    idx = 508 + (1:hold_dur1(i));
    pow_s2(:,i) = mean(tmp(idx,:),1);
    
    idx = (508 + hold_dur1(i)):size(tmp,1);
    pow_s3(:,i) = mean(tmp(idx,:),1);
end



pow_s1a = mean(pow_s1,2);
pow_s2a = mean(pow_s2,2);
pow_s3a = mean(pow_s3,2);
figure;
hold on
boxplot([pow_s1a pow_s2a pow_s3a])
hline(0,'--r')
xticks([1:3])
xticklabels({'Rest','Hold','Move'})
ylim([-1.25 3.5])
plot_beautify
ylabel('Z-score relative to rest')
xlim([1.5 3.5])


% plotting mu power
val=pow_s2a-pow_s3a;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*4;    
    if ms>=0.0 && bad_chI(j)
        e_h = el_add(elecmatrix(j,:), 'color', 'r','msize',abs(ms));
    elseif ms<0 && bad_chI(j)
        e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',abs(ms));
    end
end
set(gcf,'Color','w')
title('EC 189 Mu power (hold-move)')

% plotting hG power
tmp = mean(hg_pow,2);
figure;
imagesc(tmp(ecog_grid))
% plot on grid
val=tmp;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*8;
    c='b';
    if ms>0.5
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end

% plotting PCs first
[coeff,score,latent]=pca(aa);
tmp=coeff(:,1);
figure;
imagesc(tmp(ecog_grid))
% plot on grid
val=tmp;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = (val(j)*60);    
    if ms>=0
        c='r';
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    else
        c='b';
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end



% ERPs
figure;
hold on
ch=235;
hg_ep1 = squeeze(mean(hg_ep(:,ch,:),3));
alp_ep1 = squeeze(mean(alp_ep(:,ch,:),3));
mu_ep1 = squeeze(mean(mu_ep(:,ch,:),3));
tt=-1:(1/Fs):7;
if length(tt)>size(alp_ep1,1)
    tt=tt(1:end-1);
end
plot(tt,mean(hg_ep1,2),'b','LineWidth',1)
plot(tt,mean(mu_ep1,2),'r','LineWidth',1)
plot(tt,mean(alp_ep1,2),'k','LineWidth',1)
vline([0 3.7])
hline(0)
xlim([-1 6])
legend({'hG','narrow mu','brdband mu (8-13Hz)'})
xlabel('Time (s)')
ylabel('Z score')
plot_beautify
axis tight
title('Amplitude ERPs M1 Channel')

% phase amplitude coupling between the hG and mu at specific task phases
%bpfilt is the one for mu
close all

% phase amplitude coupling between the hG and mu at specific task phases
%bpfilt is the one for mu
hGFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',Fs);

% mu 
% bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',6,'HalfPowerFrequency2',9, ...
%     'SampleRate',Fs);

% Example: Low-pass FIR filter for LFO
bpFilt = designfilt('lowpassiir', 'FilterOrder', 4, ...
               'HalfPowerFrequency', 3, 'SampleRate', Fs);



mu = filtfilt(bpFilt,zscore(lfp));
mu = angle(hilbert(mu));

hg = filtfilt(hGFilt,zscore(lfp));
hg = abs(hilbert(hg));
hg_mu = filtfilt(bpFilt,hg);
hg_mu = angle(hilbert(hg_mu));

plv_hold=[];
plv_move=[];
for i=1:length(trial_timings)

    % hold period
    st = trial_timings(i).movement.cue(2).time; %anin
    stp = trial_timings(i).movement.cue(3).time; %anin   
    %st = st+0.5;
    %stp=st+3;
    index = (lfp_time >= st) .* (lfp_time<=stp);
    
    hg1 = hg_mu(logical(index),:);
    % hg1 = hg(logical(index),:);
    % hg1=zscore(hg1);
    % hg_mu = filtfilt(bpFilt,hg1);    
    % hg_mu = angle(hilbert(hg_mu));
    
    mu1 = mu(logical(index),:);
    % mu1=zscore(mu1);
    % mu1=angle(hilbert(mu1));
    
    tmp = circ_mean(mu1-hg1);
    plv_hold(i,:) = tmp;

    % move period
    st = trial_timings(i).movement.cue(3).time; %anin
    stp = st+3;    
    index = (lfp_time >= st) .* (lfp_time<=stp);
    
    hg1 = hg_mu(logical(index),:);
    % hg1 = hg(logical(index),:);
    % hg1=zscore(hg1);
    % hg_mu = filtfilt(bpFilt,hg1);
    % hg_mu = angle(hilbert(hg_mu));
    % 
    mu1 = mu(logical(index),:);
    % mu1=zscore(mu1);
    % mu1=angle(hilbert(mu1));
    % 
    tmp = circ_mean(mu1-hg1);
    plv_move(i,:) = tmp;
end

a = exp(1i*plv_hold);
a = mean(a,1);
a = abs(a);

b = exp(1i*plv_move);
b = mean(b,1);
b = abs(b);

figure;
boxplot([a(bad_chI)' b(bad_chI)'],'Notch','on')
xticks(1:2)
xticklabels({'Hold','Move'})
[p,h]=signrank(a(bad_chI),b(bad_chI))
ylabel('PAC')
plot_beautify
title('EC189 hG LFO PAC')

% plot on brain
val=b-a;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*16;
    c='b';
    if ms>0.0 && bad_chI(j)==1
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end
plot_beautify
title('EC 189 LFO hG PAC (hold)')
sum((b(bad_chI)'- a(bad_chI)')>0)/sum(bad_chI)


%% LOOK AT 1/F AND POWER IN STATE 3 VS. 1
%ACTUAL MOTOR CONTROL
% EC210

clc;clear
close all

% loading data
get_data_analyses_EC210
imaging_EC210
%close all

cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210')


ecog_grid=[];
k=1;
for i=1:16:256
    ecog_grid(k,:) = i:i+15;
    k=k+1;
end
ecog_grid = flipud(fliplr(ecog_grid'));

lfp=zscore(lfp);

%%% OTHER ANALYSES
% get power of LoMU in move, hold and rest states? 
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
    'HalfPowerFrequency1',4.5,'HalfPowerFrequency2',7.5, ...
    'SampleRate',Fs);

bpFilt1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',13, ...
    'SampleRate',Fs);


%ch= [204:209 219:224];
ch=1:256;

% get mu and hg
mu = [];hg=[];
alp=[];
mu = filtfilt(bpFilt,lfp(:,ch));
mu = abs(hilbert(mu));
%mu = mean(mu,2);
%mu=zscore(mu);

alp = filtfilt(bpFilt1,lfp(:,ch));
alp = abs(hilbert(alp));
%alp = mean(alp,2);
%alp=zscore(alp);

tmp_hg=[];
for j=1:8
    tmp = filtfilt(filterbank(j).b,filterbank(j).a,...
        lfp(:,ch));
    tmp = abs(hilbert(tmp));
    tmp_hg = cat(3,tmp_hg,tmp);
end
tmp_hg = squeeze(mean(tmp_hg,3));
hg=tmp_hg;
%hg = squeeze(mean(tmp_hg,2));
%hg=zscore(hg);


% get a sense of the timing of events
trial_times=[];
for i=1:length(trial_timings)
     timings = [trial_timings(i).movement.cue.time];
     trial_times(i,:) = timings;
end

trial_times
diff(trial_times')'
trial_times(2:end,1) - trial_times(1:end-1,3)

% plot
% also extract power spectrum within each signal portion (1/f)
figure;
hold on
plot(lfp_time,smooth(zscore(hg(:,140)),100))
plot(lfp_time,smooth(zscore(mu(:,140)),100))
legend({'hG','mu'})

mu_ep=[];
hg_ep=[];
alp_ep=[];
kin_ep=[];
osc_clus=[];
stats=[];
pow_freq=[];
ffreq=[];
hold_dur = [];
lfp_epochs={};
for i=1:length(trial_timings)
    timings = [trial_timings(i).movement.cue.time];
    for j=1:length(timings)
        vline(timings(1),'r') % rest
        vline(timings(2),'y') % hold
        vline(timings(3),'g') % go
    end
    go_time = trial_timings(i).movement.cue(2).time; %anin
    st = go_time-1; % go back 1s into the rest period
    stp = go_time+4.5; % average hold time is 1.7s, go period is 3s.

    hold_dur(i) = trial_timings(i).movement.cue(3).time - ...
        trial_timings(i).movement.cue(2).time;

    index = (kin_time_resample >= st) .* (kin_time_resample<=stp);
    kin_ep(:,i) = kindata_full_length_resampled(logical(index),13);

    index = (lfp_time >= st) .* (lfp_time<=stp);
    mu_ep(:,:,i) = mu(logical(index),:);
    hg_ep(:,:,i) = hg(logical(index),:);
    alp_ep(:,:,i) = alp(logical(index),:);

    % only during the go period or hold period for 1/f
    st = trial_timings(i).movement.cue(2).time; %anin
    stp = trial_timings(i).movement.cue(3).time; %anin
    %stp=st+3; % duration
    index = (lfp_time >= st) .* (lfp_time<=stp);
    data = lfp(logical(index),:);
    lfp_epochs{i} = data;

    spectral_peaks=[];
    stats_tmp=[];
    parfor ii=1:size(data,2)
        x = data(:,ii);

        [Pxx,F] = pwelch(x,512,256,512,Fs);
        %[Pxx,F] = pwelch(x,hamming(1000),500,2048,1000);
        pow_freq = [pow_freq;Pxx' ];
        ffreq = [ffreq ;F'];
        idx = logical((F>0) .* (F<=40));
        F1=F(idx);
        F1=log2(F1);
        power_spect = Pxx(idx);
        power_spect = log2(power_spect);
        tb=fitlm(F1,power_spect,'RobustOpts','on');
        stats_tmp = [stats_tmp tb.Coefficients.pValue(2)];
        bhat = tb.Coefficients.Estimate;
        x = [ones(length(F1),1) F1];
        yhat = x*bhat;

        %plot
        % figure;
        % plot(F1,power_spect,'LineWidth',1);
        % hold on
        % plot(F1,yhat,'LineWidth',1);

        % get peaks in power spectrum at specific frequencies
        power_spect = zscore(power_spect - yhat);
        [aa ,bb]=findpeaks(power_spect);
        peak_loc = bb(find(power_spect(bb)>1));
        freqs = 2.^F1(peak_loc);
        pow = power_spect(peak_loc);

        %store
        spectral_peaks(ii).freqs = freqs;
        spectral_peaks(ii).pow = pow;

    end


    % getting oscillation clusters
    if ~isempty(spectral_peaks)
        osc_clus_tmp=[];
        for f=2:40
            ff = [f-1 f+1];
            tmp=0;ch_tmp=[];
            for j=1:length(spectral_peaks)
                freqs = spectral_peaks(j).freqs;
                for k=1:length(freqs)
                    if ff(1) <= freqs(k)  && freqs(k) <= ff(2)
                        tmp=tmp+1;
                        ch_tmp = [ch_tmp j];
                    end
                end
            end
            osc_clus_tmp = [osc_clus_tmp tmp];
        end
        osc_clus = [osc_clus;osc_clus_tmp];
    end

end


save lfp_epochs_holdState lfp_epochs Fs bad_chI -v7.3

% plot oscillation clusters
f=2:40;
figure;
hold on
plot(f,osc_clus/sum(bad_chI),'Color',[.5 .5 .5 .5],'LineWidth',.5)
plot(f,mean(osc_clus,1)/sum(bad_chI),'b','LineWidth',2)
xlabel('Freq.')
ylabel('Prop of channels')
plot_beautify
title('1/f sig.')

% plot power spectrum
figure;
hold on
plot(ffreq(1,:),log10(pow_freq'),'Color',[.5 .5 .5 .5])
plot(ffreq(1,:),log10(mean(pow_freq,1)),'k','LineWidth',2)
xlim([0 200])

% get all the electrodes with peak between 8.0Hz and 10Hz
ch_idx=[];
for i=1:length(spectral_peaks)
    if sum(i==bad_ch)==0
        f = spectral_peaks(i).freqs;
        if sum( (f>=5) .* (f<=8) ) >= 1
            ch_idx=[ch_idx i];
        end
    end
end
length(ch_idx)/sum(bad_chI)
I = zeros(256,1);
I(ch_idx)=1;
%ecog_grid = TrialData.Params.ChMap
figure;imagesc(I(ecog_grid))
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'rh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
I = logical(I);
e_h = el_add(elecmatrix([I],:), 'color', 'b', 'msize',8);


m = mean(mu_ep(1:254*2,:,:),1);
s = std(mu_ep(1:254*2,:,:),1);
mu_ep = (mu_ep-m)./s;

m = mean(hg_ep(1:254*2,:,:),1);
s = std(hg_ep(1:254*2,:,:),1);
hg_ep = (hg_ep-m)./s;

m = mean(alp_ep(1:254*2,:,:),1);
s = std(alp_ep(1:254*2,:,:),1);
alp_ep = (alp_ep-m)./s;


% getting average hG power over grid 
dur = diff(trial_times')';
hold_dur1 = round(Fs*hold_dur');% in samples, duration of hold
hg_pow=[];
pow_s1=[];
pow_s2=[];
pow_s3=[];
aa=[];len=[];
for i=1:length(hold_dur1)
    %tmp = squeeze(mu_ep(:,:,i)); 
    tmp = squeeze(hg_ep(:,:,i)); 
    % first 1s or 508 samples is rest (baseline)
    %idx = 509: (509+hold_dur1(i));
    idx = (509+hold_dur1(i)):size(tmp,1);
    %hg_pow(:,i) = mean(tmp(idx,:),1);
    hg_pow(:,i) = mean(tmp(idx,:));
    aa=[aa;tmp(idx,:)];
    len=[len;length(idx)];

    
    % states 1,2,3, mu
    tmp = squeeze(mu_ep(:,:,i)); 
    idx=1:508;
    pow_s1(:,i) = mean(tmp(idx,:),1);
    
    idx = 508 + (1:hold_dur1(i));
    pow_s2(:,i) = mean(tmp(idx,:),1);
    
    idx = (508 + hold_dur1(i)):size(tmp,1);
    pow_s3(:,i) = mean(tmp(idx,:),1);
end


pow_s1a = mean(pow_s1,2);
pow_s2a = mean(pow_s2,2);
pow_s3a = mean(pow_s3,2);
figure;
hold on
boxplot([pow_s1a pow_s2a pow_s3a])
hline(0,'--r')
xticks([1:3])
xticklabels({'Rest','Hold','Move'})
ylim([-2 2])
plot_beautify
ylabel('Z-score relative to rest')
xlim([1.5 3.5])
title('EC 210 Mu power')

% plotting mu power
val=pow_s2a-pow_s3a;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'rh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*4;    
    if ms>=0.0 && bad_chI(j)
        e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',abs(ms));
    % elseif ms<0 && bad_chI(j)
    %     e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',abs(ms));
    % end
    end
end
set(gcf,'Color','w')
title('EC 210 Mu power (hold-move)')
sum(val(bad_chI)>0)/sum(bad_chI)

% hg power
tmp = mean(hg_pow,2);
figure;
imagesc(tmp(ecog_grid))
% plot on grid
val=tmp;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'rh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*10;
    c='b';
    if ms>0.0
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end

% plotting PCs first
[coeff,score,latent]=pca(aa);
tmp=coeff(:,2);
figure;
imagesc(tmp(ecog_grid))
% plot on grid
val=tmp;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'rh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = (val(j)*60);    
    if ms>=0
        c='r';
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    else
        c='b';
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end



% ERPs
figure;
hold on
ch=74;
hg_ep1 = squeeze((hg_ep(:,ch,:)));
alp_ep1 = squeeze((alp_ep(:,ch,:)));
mu_ep1 = squeeze((mu_ep(:,ch,:)));
tt=-1:(1/Fs):4.5;
if length(tt)>size(alp_ep1,1)
    tt=tt(1:end-1);
end
plot(tt,mean(hg_ep1,2),'b','LineWidth',1)
plot(tt,mean(mu_ep1,2),'r','LineWidth',1)
plot(tt,mean(alp_ep1,2),'k','LineWidth',1)
vline([0 1.7])
hline(0)
xlim([-1 4.5])
legend({'hG','narrow beta','beta'})
xlabel('Time (s)')
ylabel('Z score')
plot_beautify
axis tight

close all

% phase amplitude coupling between the hG and mu at specific task phases
%bpfilt is the one for mu
hGFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',Fs);

% mu 
% bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',5.5,'HalfPowerFrequency2',8.5, ...
%     'SampleRate',Fs);

% Example: Low-pass FIR filter for LFO
bpFilt = designfilt('lowpassiir', 'FilterOrder', 4, ...
               'HalfPowerFrequency', 3, 'SampleRate', Fs);



mu = filtfilt(bpFilt,zscore(lfp));
mu = angle(hilbert(mu));

hg = filtfilt(hGFilt,zscore(lfp));
hg = abs(hilbert(hg));
hg_mu = filtfilt(bpFilt,hg);
hg_mu = angle(hilbert(hg_mu));

plv_hold=[];
plv_move=[];
for i=1:length(trial_timings)

    % hold period
    st = trial_timings(i).movement.cue(2).time; %anin
    stp = trial_timings(i).movement.cue(3).time; %anin   
    %st = st+0.5;
    %stp=st+3;
    index = (lfp_time >= st) .* (lfp_time<=stp);
    
    hg1 = hg_mu(logical(index),:);
    % hg1 = hg(logical(index),:);
    % hg1=zscore(hg1);
    % hg_mu = filtfilt(bpFilt,hg1);    
    % hg_mu = angle(hilbert(hg_mu));
    
    mu1 = mu(logical(index),:);
    % mu1=zscore(mu1);
    % mu1=angle(hilbert(mu1));
    
    tmp = circ_mean(mu1-hg1);
    plv_hold(i,:) = tmp;

    % move period
    st = trial_timings(i).movement.cue(3).time; %anin
    stp = st+3;    
    index = (lfp_time >= st) .* (lfp_time<=stp);
    
    hg1 = hg_mu(logical(index),:);
    % hg1 = hg(logical(index),:);
    % hg1=zscore(hg1);
    % hg_mu = filtfilt(bpFilt,hg1);
    % hg_mu = angle(hilbert(hg_mu));
    % 
    mu1 = mu(logical(index),:);
    % mu1=zscore(mu1);
    % mu1=angle(hilbert(mu1));
    % 
    tmp = circ_mean(mu1-hg1);
    plv_move(i,:) = tmp;
end

a = exp(1i*plv_hold);
a = mean(a,1);
a = abs(a);

b = exp(1i*plv_move);
b = mean(b,1);
b = abs(b);

figure;
boxplot([a(bad_chI)' b(bad_chI)'],'Notch','on')
xticks(1:2)
xticklabels({'Hold','Move'})
[p,h]=signrank(a(bad_chI),b(bad_chI))
title('EC 210')
ylabel('PAC')
plot_beautify

% plot on brain
val=a;
figure;
good_ch = find(bad_chI==1);
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'rh',1,1,1);
e_h = el_add(elecmatrix([good_ch],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*8;
    c='b';
    if ms>0.0 && bad_chI(j)==1
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end
plot_beautify
title('EC 210 mu hG PAC (hold)')
sum((b(bad_chI)'- a(bad_chI)')>0)/sum(bad_chI)




%% %% LOOK AT 1/F AND POWER IN STATE 3 VS. 1
%  EC176




clc;clear
close all

% loading data
get_data_analyses_EC176
imaging_EC176
%close all

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

%%% OTHER ANALYSES
% get power of LoMU in move, hold and rest states? 
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
    'HalfPowerFrequency1',11,'HalfPowerFrequency2',17, ...
    'SampleRate',Fs);

bpFilt1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',13,'HalfPowerFrequency2',30, ...
    'SampleRate',Fs);


%ch= [204:209 219:224];
ch=1:256;

% get mu and hg
mu = [];hg=[];
alp=[];
mu = filtfilt(bpFilt,lfp(:,ch));
mu = abs(hilbert(mu));
%mu = mean(mu,2);
%mu=zscore(mu);

alp = filtfilt(bpFilt1,lfp(:,ch));
alp = abs(hilbert(alp));
%alp = mean(alp,2);
%alp=zscore(alp);

tmp_hg=[];
parfor j=1:8
    tmp = filtfilt(filterbank(j).b,filterbank(j).a,...
        lfp(:,ch));
    tmp = abs(hilbert(tmp));
    tmp_hg(j,:,:) = (tmp);
end
tmp_hg = squeeze(mean(tmp_hg,1));
hg=tmp_hg;
%hg = squeeze(mean(tmp_hg,2));
%hg=zscore(hg);


% get a sense of the timing of events
trial_times=[];
for i=1:length(trial_timings)
     timings = [trial_timings(i).movement.cue.time];
     trial_times(i,:) = timings;
end

trial_times
diff(trial_times')'
trial_times(2:end,1) - trial_times(1:end-1,3)

% plot
% also extract power spectrum within each signal portion (1/f)
figure;
hold on
plot(lfp_time,smooth(zscore(hg(:,247)),100))
plot(lfp_time,smooth(zscore(mu(:,247)),100))
legend({'hG','mu'})

mu_ep=[];
hg_ep=[];
alp_ep=[];
kin_ep=[];
osc_clus=[];
stats=[];
pow_freq=[];
ffreq=[];
hold_dur = [];
pow_move_trial=[];
pow_hold_trial=[];
lfp_epochs={};
for i=1:length(trial_timings)
    timings = [trial_timings(i).movement.cue.time];
    for j=1:length(timings)
        vline(timings(1),'r') % rest
        vline(timings(2),'y') % hold
        vline(timings(3),'g') % go
    end
    go_time = trial_timings(i).movement.cue(2).time; %anin
    st = go_time-1; % go back 1s into the rest period
    stp = go_time+7; % average hold time is 1.7s, go period is 3s.

    hold_dur(i) = trial_timings(i).movement.cue(3).time - ...
        trial_timings(i).movement.cue(2).time;

    index = (kin_time_resample >= st) .* (kin_time_resample<=stp);
    kin_ep(:,i) = kindata_full_length_resampled(logical(index),13);

    index = (lfp_time >= st) .* (lfp_time<=stp);
    mu_ep(:,:,i) = mu(logical(index),:);
    hg_ep(:,:,i) = hg(logical(index),:);
    alp_ep(:,:,i) = alp(logical(index),:);

    % only during the go period or hold period for 1/f
    st = trial_timings(i).movement.cue(2).time; %anin
    stp = trial_timings(i).movement.cue(3).time; %anin
    %stp=st+3; % duration
    index = (lfp_time >= st) .* (lfp_time<=stp);
    data = lfp(logical(index),:);

    lfp_epochs{i}=data;

    spectral_peaks=[];
    stats_tmp=[];
    parfor ii=1:size(data,2)
        x = data(:,ii);

        [Pxx,F] = pwelch(x,512,256,512,Fs);
        %[Pxx,F] = pwelch(x,hamming(1000),500,2048,1000);
        if bad_chI(ii)
            pow_freq = [pow_freq;Pxx' ];
            ffreq = [ffreq ;F'];
        end
        idx = logical((F>=2) .* (F<=40));
        F1=F(idx);
        F1=log2(F1);
        power_spect = Pxx(idx);
        power_spect = log2(power_spect);
        tb=fitlm(F1,power_spect,'RobustOpts','on');
        stats_tmp = [stats_tmp tb.Coefficients.pValue(2)];
        bhat = tb.Coefficients.Estimate;
        x = [ones(length(F1),1) F1];
        yhat = x*bhat;

        %plot
        % figure;
        % plot(F1,power_spect,'LineWidth',1);
        % hold on
        % plot(F1,yhat,'LineWidth',1);

        % get peaks in power spectrum at specific frequencies
        power_spect = zscore(power_spect - yhat);
        [aa ,bb]=findpeaks(power_spect);
        peak_loc = bb(find(power_spect(bb)>1));
        freqs = 2.^F1(peak_loc);
        pow = power_spect(peak_loc);

        %store
        spectral_peaks(ii).freqs = freqs;
        spectral_peaks(ii).pow = pow;

    end


    % getting oscillation clusters
    if ~isempty(spectral_peaks)
        osc_clus_tmp=[];
        for f=2:40
            ff = [f-1 f+1];
            tmp=0;ch_tmp=[];
            for j=1:length(spectral_peaks)
                if bad_chI(j)
                    freqs = spectral_peaks(j).freqs;
                    for k=1:length(freqs)
                        if ff(1) <= freqs(k)  && freqs(k) <= ff(2)
                            tmp=tmp+1;
                            ch_tmp = [ch_tmp j];
                        end
                    end
                end
            end
            osc_clus_tmp = [osc_clus_tmp tmp];
        end
        osc_clus = [osc_clus;osc_clus_tmp];
    end

    % power spectrum on hold vs. move periods
    st = trial_timings(i).movement.cue(2).time; %anin
    stp = trial_timings(i).movement.cue(3).time; %anin    
    index = (lfp_time >= st) .* (lfp_time<=stp);
    data = lfp(logical(index),:);
    pow_hold=[];
    parfor ii=1:size(data,2)
        x = data(:,ii);
        [Pxx,F] = pwelch(x,512,256,512,Fs);        
        if bad_chI(ii)
            pow_hold = [pow_hold;Pxx' ];            
        end
    end
    pow_hold_trial(i,:) = mean(log(pow_hold),1);

    st = trial_timings(i).movement.cue(3).time; %anin
    stp = st+3;
    index = (lfp_time >= st) .* (lfp_time<=stp);
    data = lfp(logical(index),:);
    pow_move=[];
    parfor ii=1:size(data,2)
        x = data(:,ii);
        [Pxx,F] = pwelch(x,512,256,512,Fs);        
        if bad_chI(ii)
            pow_move = [pow_move;Pxx' ];            
        end
    end
    pow_move_trial(i,:) = mean(log(pow_move),1);


end

%save lfp_epochs_holdState lfp_epochs Fs bad_chI -v7.3

% plot move and hold power spectra
figure;hold on
plot(ffreq(1,:),mean(pow_move_trial,1))
plot(ffreq(1,:),mean(pow_hold_trial,1))
legend('Move','Hold')
figure;
plot(ffreq(1,:),mean(pow_hold_trial,1)-mean(pow_move_trial,1))

% plot oscillation clusters
f=2:40;
figure;
hold on
plot(f,osc_clus/sum(bad_chI),'Color',[.5 .5 .5 .5],'LineWidth',.5)
plot(f,mean(osc_clus,1)/sum(bad_chI),'b','LineWidth',2)
xlabel('Freq.')
ylabel('Prop of channels')
plot_beautify
title('1/f sig.')

% plot power spectrum
figure;
hold on
plot(ffreq(1,:),log(pow_freq'),'Color',[.5 .5 .5 .5])
plot(ffreq(1,:),(mean(log(pow_freq),1)),'k','LineWidth',2)
xlim([0 200])

% get all the electrodes with peak between 8.0Hz and 10Hz
ch_idx=[];
for i=1:length(spectral_peaks)
    if sum(i==bad_ch)==0
        f = spectral_peaks(i).freqs;
        if sum( (f>=13) .* (f<=18) ) >= 1
            ch_idx=[ch_idx i];
        end
    end
end
length(ch_idx)/sum(bad_chI)
I = zeros(256,1);
I(ch_idx)=1;
%ecog_grid = TrialData.Params.ChMap
figure;imagesc(I(ecog_grid))
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
I = logical(I);
e_h = el_add(elecmatrix([I],:), 'color', 'b', 'msize',8);


m = mean(mu_ep(1:254*2,:,:),1);
s = std(mu_ep(1:254*2,:,:),1);
mu_ep = (mu_ep-m)./s;

m = mean(hg_ep(1:254*2,:,:),1);
s = std(hg_ep(1:254*2,:,:),1);
hg_ep = (hg_ep-m)./s;

m = mean(alp_ep(1:254*2,:,:),1);
s = std(alp_ep(1:254*2,:,:),1);
alp_ep = (alp_ep-m)./s;


% getting average hG power over grid 
dur = diff(trial_times')';
hold_dur1 = round(Fs*hold_dur');% in samples, duration of hold
hg_pow=[];
pow_s1=[];
pow_s2=[];
pow_s3=[];
aa=[];len=[];
for i=1:length(hold_dur1)
    %tmp = squeeze(mu_ep(:,:,i)); 
    tmp = squeeze(hg_ep(:,:,i)); 
    % first 1s or 508 samples is rest (baseline)
    %idx = 509: (509+hold_dur1(i));
    idx = (509+hold_dur1(i)):size(tmp,1);
    %hg_pow(:,i) = mean(tmp(idx,:),1);
    hg_pow(:,i) = mean(tmp(idx,:));
    aa=[aa;tmp(idx,:)];
    len=[len;length(idx)];

    
    % states 1,2,3, mu
    tmp = squeeze(mu_ep(:,:,i)); 
    idx=1:508;
    pow_s1(:,i) = mean(tmp(idx,:),1);
    
    idx = 508 + (1:hold_dur1(i));
    pow_s2(:,i) = mean(tmp(idx,:),1);
    
    idx = (508 + hold_dur1(i)):size(tmp,1);
    pow_s3(:,i) = mean(tmp(idx,:),1);
end




pow_s1a = mean(pow_s1,2);
pow_s2a = mean(pow_s2,2);
pow_s3a = mean(pow_s3,2);
figure;
hold on
boxplot([pow_s1a pow_s2a pow_s3a])
hline(0,'--r')
xticks([1:3])
xticklabels({'Rest','Hold','Move'})
ylim([-1 1])
plot_beautify
ylabel('Z-score relative to rest')
xlim([1.5 3.5])
title('EC 176 Mu power')

% plotting mu power
val=pow_s2a-pow_s3a;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
good_ch=find(bad_chI==1);
e_h = el_add(elecmatrix([good_ch],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*8;    
    if ms>=0.0 && bad_chI(j)
        e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',abs(ms));
    % elseif ms<0 && bad_chI(j)
    %     e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',abs(ms));
    % end
    end
end
set(gcf,'Color','w')
title('EC 176 Mu power (hold-move)')
sum(val(bad_chI)>0)/sum(bad_chI)

% plotting hG power
tmp = mean(hg_pow,2);
%tmp = pow_s2a;
figure;
imagesc(tmp(ecog_grid))
% plot on grid
val=tmp;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*10;
    c='b';
    if ms>0.0 && bad_chI(j)
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end

% plotting PCs first
[coeff,score,latent]=pca(aa);
tmp=coeff(:,2);
figure;
imagesc(tmp(ecog_grid))
% plot on grid
val=tmp;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix([1:256],:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = (val(j)*60);    
    if ms>=0
        c='r';
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    else
        c='b';
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end



% ERPs
figure;
hold on
ch=106;
hg_ep1 = squeeze((hg_ep(:,ch,:)));
alp_ep1 = squeeze((alp_ep(:,ch,:)));
mu_ep1 = squeeze((mu_ep(:,ch,:)));
tt=-1:(1/Fs):7;
if length(tt)>size(alp_ep1,1)
    tt=tt(1:end-1);
end
plot(tt,mean(hg_ep1,2),'b','LineWidth',1)
plot(tt,mean(mu_ep1,2),'r','LineWidth',1)
plot(tt,mean(alp_ep1,2),'k','LineWidth',1)
vline([0 4])
hline(0)
xlim([-1 7])
legend({'hG','narrow beta','beta'})
xlabel('Time (s)')
ylabel('Z score')
plot_beautify
axis tight


close all

% phase amplitude coupling between the hG and mu at specific task phases
%bpfilt is the one for mu
hGFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',Fs);

% mu 
% bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',11,'HalfPowerFrequency2',17, ...
%     'SampleRate',Fs);

% Example: Low-pass FIR filter for LFO
bpFilt = designfilt('lowpassiir', 'FilterOrder', 4, ...
               'HalfPowerFrequency', 3, 'SampleRate', Fs);



mu = filtfilt(bpFilt,zscore(lfp));
mu = angle(hilbert(mu));

hg = filtfilt(hGFilt,zscore(lfp));
hg = abs(hilbert(hg));
hg_mu = filtfilt(bpFilt,hg);
hg_mu = angle(hilbert(hg_mu));

plv_hold=[];
plv_move=[];
for i=1:length(trial_timings)

    % hold period
    st = trial_timings(i).movement.cue(2).time; %anin
    stp = trial_timings(i).movement.cue(3).time; %anin   
    %st = st+0.5;
    %stp=st+3;
    index = (lfp_time >= st) .* (lfp_time<=stp);
    
    hg1 = hg_mu(logical(index),:);
    % hg1 = hg(logical(index),:);
    % hg1=zscore(hg1);
    % hg_mu = filtfilt(bpFilt,hg1);    
    % hg_mu = angle(hilbert(hg_mu));
    
    mu1 = mu(logical(index),:);
    % mu1=zscore(mu1);
    % mu1=angle(hilbert(mu1));
    
    tmp = circ_mean(mu1-hg1);
    plv_hold(i,:) = tmp;

    % move period
    st = trial_timings(i).movement.cue(3).time; %anin
    stp = st+3;    
    index = (lfp_time >= st) .* (lfp_time<=stp);
    
    hg1 = hg_mu(logical(index),:);
    % hg1 = hg(logical(index),:);
    % hg1=zscore(hg1);
    % hg_mu = filtfilt(bpFilt,hg1);
    % hg_mu = angle(hilbert(hg_mu));
    % 
    mu1 = mu(logical(index),:);
    % mu1=zscore(mu1);
    % mu1=angle(hilbert(mu1));
    % 
    tmp = circ_mean(mu1-hg1);
    plv_move(i,:) = tmp;
end

a = exp(1i*plv_hold);
a = mean(a,1);
a = abs(a);

b = exp(1i*plv_move);
b = mean(b,1);
b = abs(b);

figure;
boxplot([a(bad_chI)' b(bad_chI)'],'Notch','on')
xticks(1:2)
xticklabels({'Hold','Move'})
[p,h]=signrank(a(bad_chI),b(bad_chI))
title('EC 176')
ylabel('PAC')
plot_beautify

% plot on brain
val=a-b;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix(good_ch,:), 'color', 'w', 'msize',2);
for j=1:length(val)
    ms = val(j)*10;
    c='b';
    if val(j)>0.0 && bad_chI(j)==1
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end
plot_beautify
title('EC 179 LFO hG PAC (hold)')
sum( (b(bad_chI) - a(bad_chI) > 0 ))/sum(bad_chI)



