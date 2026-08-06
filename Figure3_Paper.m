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


%% Extracting power in mu across sessions b3 hand
% AND ALSO ERPs

% get all closed-loop trials for B3 hand. Each day, compute average power
% across trials per channel during state 3. This gives boxplot per day.

% load
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
load B3_waves_Hand_stability_hgFilterBank_PLV_AccStatsCL_v2_PLVDelta


% at each channel, get the average mu power across trials for each session
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

state_pow_days_ol={};
state_pow_days_cl={};
erps={};
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


    [state_pow,erps,acc,lfp_epochs] = get_state_pow(files,bpFilt);
    %title(['Day ' num2str(days) ' CL'])
    state_pow_days_cl{days}=state_pow;
  
end


% lfp_epochs is a cell with each cell TXCh data matrix
Fs=1e3;
bad_chI = ones(256,1);
bad_chI([108 113 118])=0;
save lfp_epochs_BCI lfp_epochs Fs bad_chI -v7.3 


%save mu_state_power_B3 state_pow_days_cl state_pow_days_ol -v7.3

% plot ERPs from a day with high mu
% think you will have to renormalize here to the first 500ms or so and then
% ensure that the y mag does not cross boxplot values
erp=[];
parfor i=1:length(erps)
    tmp = erps{i};
    tmp = tmp(501:end,:); % removing the first 500ms. 
    tmp = detrend(tmp);
    for j=1:size(tmp,2)
        tmp(:,j) = smooth(tmp(:,j),10);
    end
    m = mean(tmp(1:500,:),1);
    s = std(tmp(1:500,:),1);
    tmp = (tmp-m)./1;
    %tmp = tmp(:,ecog_grid);
    
    % earlier with full data
    % first 1800ms are basically rest plus cue
    %tmp = tmp(1:3800,:);

    % after removing first 500ms and re zscoring 
    % first 500ms is state 1, then 800ms if state 2 i.e, 1300ms
    % take the next 3s of data ie 1300+3000 = 4300ms
    
    tmp = tmp(1:4300,:);
    erp(i,:,:) = tmp;
end

x = squeeze(mean(erp,1));
figure;plot(mean(x,2));
xlim([500 4800])
xx = mean(x(1301:end,:),1);

figure;imagesc(xx(ecog_grid))

% get ERPs in a few channels
figure
ch=[116 201];
for i=1:length(ch)
    subplot(2,1,i)
    mu_ep1 = squeeze((erp(:,:,ch(i))))';
    tt = linspace(-1.3,3,size(mu_ep1,1));
    m = mean(mu_ep1,2);
    mb = sort(bootstrp(1000,@mean,mu_ep1'));
    [fillhandle,msg]=jbfill(tt,(mb(25,:)),(mb(975,:))...
        ,[0.2 0.2 0.8],[0.2 0.2 0.8],1,.25);
    hold on
    plot(tt,m,'Color','b','LineWidth',1)
    xlim([-1.3 2.5])
    %xticks([-1.3 -0.8:0.8:2.5])
    xticks([-1:0.5:2.5])
    ylim([-0.75 2.5])
    vline([-0.8 0],'--k')
    hline(0)    
    xlabel('Time (s)')
    ylabel('Z score')
    plot_beautify    
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
boxplot(pow,'Whisker',2)
%boxplot(pow)
ylabel('Z-score')
title('Mu power during BCI control')
xlabel('Days')
xticks(1:10)
plot_beautify
hline(0)
ylim([-.201 1.701])
yticks([-.2:.4:1.8])

% splitting early vs late
early_pow = pow(:,1:5);
late_pow = pow(:,6:end);
figure;
boxplot([early_pow(:) late_pow(:)],'Whisker',2)
ylabel('Z-score')
xticks(1:2)
xticklabels({'1st 5 Days','2nd 5 Days'})
plot_beautify
hline(0)
ylim([-.201 1.701])
yticks([-.2:.4:1.8])

%% stuff to save the data

set(gcf,'PaperPositionMode','auto');
print(gcf,'Mu_Power_Days_B3_Hand_AllChannels.svg','-dsvg','-painters','-r300');



