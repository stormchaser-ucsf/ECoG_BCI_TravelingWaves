%% FIGURE 4
% GOAL HERE IS TO LOAD THE PAC AND MU STATE POWER IN ARROW TASKS
% COMPARE MU HG PAC, LFO HG PAC AND SHOW THESE OSCILLATORY DYNAMICS AT THE
% TRIAL LEVEL AFTER SHOWING THE CONFUSION MATRICES


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


%% ERPs of Mu and hG and LFO


%% Percent sig Mu-hG PAC channels 
% contrast with LFO-hG PAC

% load B1 data
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
cd(root_path)
load('ECOG_Grid_8596_000067_B3.mat')


a = load('PAC_B1_LFO_hG_rawValues_New.mat'); % LFO-hG PAC
a = load('PAC_B1_Mu_hG_rawValues_New.mat'); % Mu-hG PAC



%% Decoding relationship w/ Mahab Dist
% B1, B6 
clc;clear;
close all
subj='B6';

if strcmp(subj,'B1')


    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
    cd(root_path)
    %load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))

    hilbert_flag=1;

    imaging_B3_waves;
    load('ECOG_Grid_8596_000067_B3.mat')
    close all

    folders={'20240515', '20240517', '20240614', ...
        '20240619', '20240621', '20240626',...
        '20240710','20240712','20240731'};
    num_targets=7;
    folders=folders(1:end-1);
    cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')

elseif strcmp(subj,'B6')


    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/';
    cd(root_path)
    %load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))

    hilbert_flag=1;

    imaging_B3_waves;
    load('ECOG_Grid_8596_000067_B3.mat')
    close all

      folders = {'20250530','20250610','20250624','20250703','20250708','20250717',...
        '20250917','20250924','20251203','20251204','20251210','20260116'};
  
    num_targets=7;
end


%
d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3); % 8 to 10 or 0.5 to 5

%
% d1 = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
%     'SampleRate',1e3); % 8 to 10 or 0.5 to 5
% %
d1a = designfilt('lowpassiir', 'FilterOrder', 4, ...
    'HalfPowerFrequency', 3, 'SampleRate', 1e3);

% d1a = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',2.5, ...
%     'SampleRate',1e3); % B6 it is 0.5 to 2.5


d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);



pac_ol=[];pval_ol=[];
pac_cl=[];pval_cl=[];
pac_batch=[];pval_batch=[];
rboot_ol=[];rboot_cl=[];rboot_batch=[];
pac_raw_values={};k=1;
pac_raw_values_LFO={};
mahab_dist_days=[];
stats_mu=[];
stats_lfo=[];
bhat_mu=[];
bhat_lfo=[];

%imaging_B1_253;
%close all
tic
for i=1:length(folders)


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

    len = min(120,length(files));
    idx=randperm(length(files),len);
    files=files(idx);

    % %get the phase locking value
    % if length(files)>0
    %     disp(['Processing Day ' num2str(i) ' OL'])
    %     [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    % else
    %     pac=[];
    % end
    pac=[];


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

    len = min(120,length(files));
    idx=randperm(length(files),len);
    files=files(idx);

    % get the phase locking value
    if length(files)>0
        disp(['Processing Day ' num2str(i) ' CL'])
        % mu
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
        % lfo
        [pac1,alpha_phase1,hg_alpha_phase1] = compute_pac(files,d1a,d2);

        % run permutation test and get pvalue for each channel
        %[pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
        % [pfdr,pp]=fdr(pval,0.05);
        % sum(pval<=pfdr)/253;
        % r=abs(mean(pac,1));
        % median(r);

        % get mahab distances
        [mahab_dist] = get_mahab_dist_7DoF(files);
        mahab_dist_days(i,:) = mahab_dist;

        % get correlations
        % mu-hg pac
        x= mahab_dist';
        y = abs(mean(pac))';
        x = [ones(size(x,1),1) x];
        %[B,BINT,R,RINT,STATS1] = regress(y,x);
        mdl = fitlm(x(:,2),y,'RobustOpts','on');
        B = mdl.Coefficients.Estimate;
        stats_mu = [stats_mu mdl.Coefficients.pValue];
        bhat_mu  = [bhat_mu B];

        % lfo-hg PAC
        x= mahab_dist';
        y = abs(mean(pac1))';
        x = [ones(size(x,1),1) x];
        %[B,BINT,R,RINT,STATS1] = regress(y,x);
        mdl1 = fitlm(x(:,2),y,'RobustOpts','on');
        B1 = mdl1.Coefficients.Estimate;
        stats_lfo = [stats_lfo mdl1.Coefficients.pValue];
        bhat_lfo  = [bhat_lfo B1];
    else
        pac=[];
        rboot=[];
    end



    %sum(pac_r>0.3)/253
    % pval_cl(i,:) = pval;
    pac_cl(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    %pac_raw_values(k).boot = rboot;
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


figure;boxplot([bhat_mu(2,:)' bhat_lfo(2,:)'])
hline(0)
figure;plot(bhat_lfo(2,:),'.','MarkerSize',15)
hline(0)
title('LFO hg PAC')
ylabel('Slope')
figure;plot(bhat_mu(2,:),'.','MarkerSize',15)
hline(0)
title('mu hg PAC')
ylabel('Slope')

x= (1:size(bhat_lfo,2))';
y = bhat_lfo(2,:)';
x = [ones(size(x,1),1) x];
%[B,BINT,R,RINT,STATS1] = regress(y,x);
mdl1 = fitlm(x(:,2),y,'RobustOpts','on');

sum(stats_lfo(2,:)<=0.05)


save PAC_DecodingRelationship_B6_ArrowTask -v7.3

%% Decoding relationship w/ Mahab Dist
% B3 Arrow task 

clc;clear
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
cd(root_path)
%load session_data_B3_Hand
load session_data_B3
load('ECOG_Grid_8596_000067_B3.mat')
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))

% 
d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3); % 8 to 10 or 0.5 to 5

% 
% d1 = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
%     'SampleRate',1e3); % 8 to 10 or 0.5 to 5
% 
% d1a = designfilt('lowpassiir', 'FilterOrder', 4, ...
%                'HalfPowerFrequency', 3, 'SampleRate', 1e3);

d1a = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',2.5, ...
    'SampleRate',1e3); % B1,B3,B6 it is 0.5 to 2.5


d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);



pac_ol=[];pval_ol=[];
pac_cl=[];pval_cl=[];
pac_batch=[];pval_batch=[];
rboot_ol=[];rboot_cl=[];rboot_batch=[];
pac_raw_values={};k=1;
pac_raw_values_LFO={};
mahab_dist_days=[];
stats_mu=[];
stats_lfo=[];
bhat_mu=[];
bhat_lfo=[];
len_days = min(11,length(session_data));

%imaging_B1_253;
%close all
tic
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

    % %get the phase locking value
    % if length(files)>0
    %     disp(['Processing Day ' num2str(i) ' OL'])
    %     [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    % else
    %     pac=[];
    % end
    pac=[];


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
        % mu
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
        % lfo
        [pac1,alpha_phase1,hg_alpha_phase1] = compute_pac(files,d1a,d2);

        % run permutation test and get pvalue for each channel
        %[pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
        % [pfdr,pp]=fdr(pval,0.05);
        % sum(pval<=pfdr)/253;
        % r=abs(mean(pac,1));
        % median(r);

        % get mahab distances
        [mahab_dist] = get_mahab_dist_7DoF(files);
        mahab_dist_days(i,:) = mahab_dist;

        % get correlations
        % mu-hg pac
        x= mahab_dist';
        y = abs(mean(pac))';
        x = [ones(size(x,1),1) x];
        %[B,BINT,R,RINT,STATS1] = regress(y,x);
        mdl = fitlm(x(:,2),y,'RobustOpts','on');
        B = mdl.Coefficients.Estimate;
        stats_mu = [stats_mu mdl.Coefficients.pValue];
        bhat_mu  = [bhat_mu B];

        % lfo-hg PAC
        x= mahab_dist';
        y = abs(mean(pac1))';
        x = [ones(size(x,1),1) x];
        %[B,BINT,R,RINT,STATS1] = regress(y,x);
        mdl1 = fitlm(x(:,2),y,'RobustOpts','on');
        B1 = mdl1.Coefficients.Estimate;
        stats_lfo = [stats_lfo mdl1.Coefficients.pValue];
        bhat_lfo  = [bhat_lfo B1];
    else
        pac=[];
        rboot=[];
    end

    

    %sum(pac_r>0.3)/253
    % pval_cl(i,:) = pval;
    pac_cl(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    %pac_raw_values(k).boot = rboot;
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


figure;boxplot([bhat_mu(2,:)' bhat_lfo(2,:)'])
hline(0)
figure;plot(bhat_lfo(2,:),'.','MarkerSize',15)
hline(0)
title('LFO hg PAC')
ylabel('Slope')
figure;plot(bhat_mu(2,:),'.','MarkerSize',15)
hline(0)
title('mu hg PAC')
ylabel('Slope')

x= (1:size(bhat_lfo,2))';
y = bhat_lfo(2,:)';
x = [ones(size(x,1),1) x];
%[B,BINT,R,RINT,STATS1] = regress(y,x);
mdl1 = fitlm(x(:,2),y,'RobustOpts','on');

sum(stats_lfo(2,:)<=0.05)

save PAC_DecodingRelationship_B3_ArrowTask -v7.3


%% PLOTTING RESULTS

