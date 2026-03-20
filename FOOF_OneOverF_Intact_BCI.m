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


%% LOOK AT 1/F AND POWER IN STATE 3 VS. 1
%BCI - B3

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3); % center freq is 8.5
imaging_B3_waves;
close all
subj='B6';


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

for days=1:len_days

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

end



%% LOOK AT 1/F AND POWER IN STATE 3 VS. 1
%ACTUAL MOTOR CONTROL
