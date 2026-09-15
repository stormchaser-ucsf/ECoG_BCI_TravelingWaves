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


%% MU State Power during BCI control Arrow Task 
% B1, B3, B6

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
b3 = load('MuStatePower_B3_Arrow.mat');

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
b1 = load('MuStatePower_B1_Arrow.mat');

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
b6 = load('MuStatePower_B6_Arrow.mat');


% b1
state_pow_days_cl = b1.state_pow_days_cl;
days=1:length(state_pow_days_cl);
pow_b1=[];
for i=1:length(days)
    tmp = state_pow_days_cl{i};
    pow_b1(:,i) = tmp(:,3);
    %pow(:,i) = tmp(:,3) - tmp(:,2);
end
%pow_b1(:,end+1:12) = NaN;
pow_b1 = mean(pow_b1,2);

% b3
state_pow_days_cl = b3.state_pow_days_cl;
days=1:length(state_pow_days_cl);
pow_b3=[];
for i=1:length(days)
    tmp = state_pow_days_cl{i};
    pow_b3(:,i) = tmp(:,3);
    %pow(:,i) = tmp(:,3) - tmp(:,2);
end
%pow_b3(:,end+1:12) = NaN;
pow_b3 = mean(pow_b3,2);

% b6
state_pow_days_cl = b6.state_pow_days_cl;
days=1:length(state_pow_days_cl);
pow_b6=[];
for i=1:length(days)
    tmp = state_pow_days_cl{i};
    pow_b6(:,i) = tmp(:,3);
    %pow(:,i) = tmp(:,3) - tmp(:,2);
end
pow_b6 = mean(pow_b6,2);

res = [pow_b1(:) pow_b3(:) pow_b6(:)];
figure;
%boxplot(res,'Symbol','')
boxplot(res)
hline(0)
ylim([-0.8 1.2])
xticks(1:3)
xticklabels({'B1','B3','B6'})
ylabel('Mu Power during BCI (z)')
yticks([-0.8:.4:1.21])
plot_beautify

[p,h,stats]=signrank(res(:,1));
[p,h]=signrank(res(:,2));
[p,h]=signrank(res(:,3));

%% Decoding relationship w/ Mahab Dist
% B1, B6 
clc;clear;
close all
subj='B1';

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
% d1a = designfilt('lowpassiir', 'FilterOrder', 4, ...
%     'HalfPowerFrequency', 3, 'SampleRate', 1e3);

d1a = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',2.5, ...
    'SampleRate',1e3); % B6 it is 0.5 to 2.5


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

    % only get the 2nd half of CL files ie., CL2
    % l = round(length(online_idx)/2);
    % online_idx = online_idx(l:end);



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

    B1


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


%save PAC_DecodingRelationship_B6_ArrowTask -v7.3

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
    %online_idx=[online_idx batch_idx batch_idx1];
    online_idx=[ batch_idx ];




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

%save PAC_DecodingRelationship_B3_ArrowTask -v7.3


%% PLOTTING RESULTS
% relationship with decoding information

clc;clear

%b1
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
b1= load('PAC_DecodingRelationship_B1_ArrowTask.mat');
b1.d1a

%b6
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
b6= load('PAC_DecodingRelationship_B6_ArrowTask.mat');
b6.d1a

%b3
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
b3 = load('PAC_DecodingRelationship_B3_ArrowTask.mat');
b3.d1a

bhat_lfo = [];
bhat_mu =[];

bhat_lfo = [bhat_lfo; b1.bhat_lfo(2,:)' ;b3.bhat_lfo(2,:)';  b6.bhat_lfo(2,1:end-1)' ];
bhat_mu = [bhat_mu; b1.bhat_mu(2,:)' ;b3.bhat_mu(2,:)';  b6.bhat_mu(2,1:end-1)' ];


figure;
boxplot([bhat_mu bhat_lfo])
hline(0)

% statistics
s1=b1.stats_mu(2,:);
s3=b3.stats_mu(2,:);
s6=b6.stats_mu(2,:);

s= [s1 s3 s6];
[pfdr,pval]=fdr(s,0.05);
sum(s<=pfdr)/length(s)

s1=b1.stats_lfo(2,:);
s3=b3.stats_lfo(2,:);
s6=b6.stats_lfo(2,:);

s= [s1 s3 s6];
[pfdr,pval]=fdr(s,0.05);
sum(s<=pfdr)/length(s)

%%% as scatter plot
% mu
b1_acc = b1.bhat_mu(2,:);
b3_acc = b3.bhat_mu(2,:);
b6_acc = b6.bhat_mu(2,1:end-1);
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;
m22 = b3_acc;
m33 = b6_acc;
x=1:3;
y=[mean(m11) mean(m22) mean(m33)];
% scatter B1 and B3 and B6 individually
figure; hold on
h=hline(median(res),'k');
h.LineWidth=3;
h.XData = [0.75 1.25];

x=(1:1) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end


x=(1:1) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end

%%%% lfo
b1_acc = b1.bhat_lfo(2,:);
b3_acc = b3.bhat_lfo(2,:);
b6_acc = b6.bhat_lfo(2,1:end-1);
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;
m22 = b3_acc;
m33 = b6_acc;
x=1:3;
y=[mean(m11) mean(m22) mean(m33)];
% scatter B1 and B3 and B6 individually
hold on
h=hline(median(res),'k');
h.LineWidth=3;
h.XData = [1.75 2.25];

x=(2) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(2) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end


x=(2) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end

ylim([-0.3 0.5])
yticks([-1:.1:1])
xlim([.5 2.5])
h=hline(0);
set(h,'LineWidth',1)
xticks ''
plot_beautify
%boxplot([bhat_mu bhat_lfo])
ylim([-.2 .5])
xticks(1:2)
xticklabels({'Mu-hG','LFO-hG'})
ylabel('Slope')


%%%% stats
% lmm
subj_idx = [ones(size(b1_acc'));2*ones(size(b3_acc'));3*ones(size(b6_acc'))];
bhat = [bhat_mu bhat_lfo];bhat=bhat(:);
oscillation_type = categorical([zeros(size(bhat_mu));ones(size(bhat_lfo))]);
subject = categorical([subj_idx;subj_idx]);
data = table(bhat,oscillation_type,subject);
glm = fitlme(data,'bhat ~ 1+(oscillation_type) + (1|subject)')

% difference with CI via bootstrap
a = bhat_lfo - bhat_mu;
ab = sort(bootstrp(1000,@mean,a));
[ab(25) mean(a) ab(975)]
% pval
stat = glm.Coefficients.tStat(2);
boot=[];
parfor i=1:1000

    oscillation_type_rand=[];
    subject_rnd = [];
    pac_rnd=[];
    for ii=1:3
        idx = find(subj_idx==ii);
        mu_tmp = bhat_mu(idx);
        lfo_tmp = bhat_lfo(idx);
        subj_tmp = subj_idx(idx);
        osc_tmp = categorical([zeros(size(mu_tmp));ones(size(lfo_tmp))]);
        osc_tmp = osc_tmp(randperm(numel(osc_tmp)));
        
        subject_rnd = [subject_rnd;[subj_tmp;subj_tmp]];
        pac_rnd = [pac_rnd;[mu_tmp;lfo_tmp]];
        oscillation_type_rand = [oscillation_type_rand;osc_tmp];
    end   
    
    data_rnd = table(pac_rnd,oscillation_type_rand,subject_rnd);
    glm_rnd = fitlme(data_rnd,...
        'pac_rnd ~ 1+(oscillation_type_rand) + (1|subject_rnd)');
    boot(i) = glm_rnd.Coefficients.tStat(2);
end

figure;hist(boot)
vline(stat)
max(1/length(boot),sum(boot>stat)/length(boot))

% WSRT
stats=[];
pval=[];
for ii=1:3
    idx = find(subj_idx==ii);
    a = bhat_mu(idx);
    b = bhat_lfo(idx);
    [p,h,s] = signrank(b,a,'method','approximate');
    stats(ii) = s.zval;
    pval(ii) = p;
end
stats
pval



%% PLOTTING RESULTS
% number of significant channels 

clc;clear

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
imaging_B1_253 % already in blackrock sorted grid numbering
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
        ms = (r_mu(j))*10;
        c='b';                
        e_h = el_add(elecmatrix(j,:), 'color', c,'msize',abs(ms));
    end
end
plot_beautify





%b6
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
b6_lfo = load('PAC_B6_LFO_hG_rawValues_New_v2.mat');
b6_mu = load('PAC_B6_Mu_hG_rawValues_New_v2_CL2.mat');
% lfo sig ch
cl_days=[2:2:length(b6_lfo.pac_raw_values)];
pac_all=[];
cl=[];
for i=1:length(cl_days)
    tmp = b6_lfo.pac_raw_values(cl_days(i)).pac;
    tmp = abs(mean(tmp));
    pac_all(i,:) = tmp;

    ptmp=b6_lfo.pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);    
    %pfdr = 0.013;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
end
lfo_cl=cl;
% mu sig ch
cl_days=[2:2:length(b6_mu.pac_raw_values)];
pac_all=[];
cl=[];
for i=1:length(cl_days)
    tmp = b6_mu.pac_raw_values(cl_days(i)).pac;
    tmp = abs(mean(tmp));
    pac_all(i,:) = tmp;

    ptmp=b6_mu.pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);    
    %pfdr = 0.013;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
end
mu_cl=cl;
figure;
boxplot([mu_cl(1:end-1)' lfo_cl(1:end-1)'])
lfo_cl_all = [lfo_cl_all;lfo_cl(1:end-1)'];
mu_cl_all = [mu_cl_all;mu_cl(1:end-1)'];
subj_idx = [subj_idx;2*ones(size(mu_cl,2)-1,1)];
[p,h]=signrank(mu_cl,lfo_cl);
title(['B6 pval ' num2str(p)])
plot_beautify
xticks(1:2)
xticklabels({'Mu-hG','LFO-hG'})
ylabel('% Sig. Channels')
yticks([0:.1:1])
ylim([0 0.31])

%b3
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
b3_lfo = load('PAC_B3_LFO_hG_rawValues_Arrow_New_v2.mat');
b3_mu = load('PAC_B3_Mu_hG_rawValues_Arrow_New_v2_CL2.mat');
% lfo sig ch
cl_days=[2:2:length(b3_lfo.pac_raw_values)];
pac_all=[];
cl=[];
for i=1:length(cl_days)
    tmp = b3_lfo.pac_raw_values(cl_days(i)).pac;
    tmp = abs(mean(tmp));
    pac_all(i,:) = tmp;

    ptmp=b3_lfo.pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);    
    %pfdr = 0.013;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
end
lfo_cl=cl;
% mu sig ch
cl_days=[2:2:length(b3_mu.pac_raw_values)];
pac_all=[];
cl=[];
for i=1:length(cl_days)
    tmp = b3_mu.pac_raw_values(cl_days(i)).pac;
    tmp = abs(mean(tmp));
    pac_all(i,:) = tmp;

    ptmp=b3_mu.pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);    
    %pfdr = 0.013;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
end
mu_cl=cl;
mu_cl_all = [mu_cl_all;mu_cl(1:end)'];
lfo_cl_all = [lfo_cl_all;lfo_cl(1:end)'];
subj_idx = [subj_idx;3*ones(size(mu_cl,2),1)];
figure;
boxplot([mu_cl(1:end)' lfo_cl(1:end)'],'Symbol','')
[p,h]=signrank(mu_cl,lfo_cl);
title(['B3 pval ' num2str(p)])
plot_beautify
xticks(1:2)
xticklabels({'Mu-hG','LFO-hG'})
ylabel('% Sig. Channels')
yticks([0:.1:1])
ylim([0 0.61])


figure;
%boxplot([mu_cl_all lfo_cl_all])
signrank(mu_cl_all,lfo_cl_all)
ylim([0 0.7])
xticks(1:2)
xticklabels({'Mu-hG', 'LFO-hG'})
ylabel('% Sig. Channels')
hold on
col = {'r','b','k'};
col1= [0.7 0 0 0.5;...
    0 0 0.7 0.5;...
    0.5 0.5 0.5 0.5];
% scatter
for i=1:3
    idx = find(subj_idx==i);
    
    tmp = mu_cl_all(idx);
    %tmp = tmp(tmp<0.25);
    aa = ones(length(tmp),1) + randn(length(tmp),1)*0.1;    
    plot(aa,tmp,'.','MarkerSize',30,'Color',col1(i,:))

    tmp1 = lfo_cl_all(idx);
    aa1 = 2*ones(length(idx),1) + randn(length(idx),1)*0.1;    
    plot(aa1,tmp1,'.','MarkerSize',30,'Color',col1(i,:))

    % plot([aa(:) aa1(:)]', [tmp(:) tmp1(:)]', '-',...
    %     'Color',col1(i,:),'LineWidth',.5)

end
xlim([.5 2.5])

%%%% as better scatter plot
% mu
b1_acc = mu_cl_all(subj_idx==1)';
b3_acc = mu_cl_all(subj_idx==3)';
b6_acc = mu_cl_all(subj_idx==2)';
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;%m11(m11>.25) = NaN;
m22 = b3_acc;%m22(m22>.25) = NaN;
m33 = b6_acc;%m33(m33>.25) = NaN;
x=1:3;
y=[mean(m11) mean(m22) mean(m33)];
% scatter B1 and B3 and B6 individually
figure; hold on
h=hline(median(res),'k');
h.LineWidth=3;
h.XData = [0.75 1.25];

x=(1:1) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(1:1) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end


x=(1:1) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end

%%%% lfo
b1_acc = lfo_cl_all(subj_idx==1)';
b3_acc = lfo_cl_all(subj_idx==3)';
b6_acc = lfo_cl_all(subj_idx==2)';
res=[b1_acc b3_acc b6_acc];
m11 = b1_acc;
m22 = b3_acc;
m33 = b6_acc;
x=1:3;
y=[mean(m11) mean(m22) mean(m33)];
% scatter B1 and B3 and B6 individually
hold on
h=hline(median(res),'k');
h.LineWidth=3;
h.XData = [1.75 2.25];

x=(2) + 0.1*randn(length(m11),1);
h=scatter(x,[m11],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.3;
end

x=(2) + 0.1*randn(length(m22),1);
h=scatter(x,[m22],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.3;
end


x=(2) + 0.1*randn(length(m33),1);
h=scatter(x,[m33],70,'filled');
for i=1:1
    h(i).MarkerFaceColor = 'k';
    h(i).MarkerFaceAlpha = 0.3;
end

ylim([-0.3 0.5])
yticks([-1:.1:1])
xlim([.5 2.5])
h=hline(0);
set(h,'LineWidth',1)
xticks ''
plot_beautify

%boxplot([mu_cl_all lfo_cl_all])
ylim([0 0.6])
xticks(1:2)
xticklabels({'Mu-hG','LFO-hG'})
ylabel('% Sig. PAC Chan.')

% stats
% LME nonparametric
% PAC ~ 1 + Oscillation_Type + (1|subject)
pac = [mu_cl_all;lfo_cl_all];
oscillation_type = categorical([zeros(size(mu_cl_all));ones(size(lfo_cl_all))]);
subject = categorical([subj_idx;subj_idx]);
data = table(pac,oscillation_type,subject);
glm = fitlme(data,'pac ~ 1+(oscillation_type) + (1|subject)')

% pval
stat = glm.Coefficients.tStat(2);
boot=[];
parfor i=1:1000

    oscillation_type_rand=[];
    subject_rnd = [];
    pac_rnd=[];
    for ii=1:3
        idx = find(subj_idx==ii);
        mu_tmp = mu_cl_all(idx);
        lfo_tmp = lfo_cl_all(idx);
        subj_tmp = subj_idx(idx);
        osc_tmp = categorical([zeros(size(mu_tmp));ones(size(lfo_tmp))]);
        osc_tmp = osc_tmp(randperm(numel(osc_tmp)));
        
        subject_rnd = [subject_rnd;[subj_tmp;subj_tmp]];
        pac_rnd = [pac_rnd;[mu_tmp;lfo_tmp]];
        oscillation_type_rand = [oscillation_type_rand;osc_tmp];
    end   
    
    data_rnd = table(pac_rnd,oscillation_type_rand,subject_rnd);
    glm_rnd = fitlme(data_rnd,...
        'pac_rnd ~ 1+(oscillation_type_rand) + (1|subject_rnd)');
    boot(i) = glm_rnd.Coefficients.tStat(2);
end

figure;hist(boot)
vline(stat)
max(1/length(boot),sum(boot>stat)/length(boot))

% WSRT
stats=[];
pval=[];
for ii=1:3
    idx = find(subj_idx==ii);
    a = mu_cl_all(idx);
    b = lfo_cl_all(idx);
    [p,h,s] = signrank(a,b,'method','approximate');
    stats(ii) = s.zval;
    pval(ii) = p;
end


%% LOAD MAT FILES AND PLOT RESULTS
clc;clear
cd('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\mat_plots')
uiopen('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\mat_plots\B3Hand_DecodingAcc_EarlyVsLateDays.fig',1)
ylim([40 100])
ylabel('Decoding Accuracy')

% svg
set(gcf,'PaperPositionMode','auto');
print(gcf,'Slopes_ArrowTask_PAC_LFOhG__MuhG_DecodingInfo_AllDays_New.svg','-dsvg','-painters','-r300');
    

% png
set(gcf,'PaperPositionMode','auto');
print(gcf,'B1_Arrow_LFO_hG_PAC.png','-dpng','-r500');
