%% FIGURE 1 OF THE PAPER
% show the ecog grid
% task layout
% performance in the hand task over days

clc
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))
cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves')

% load B1 253
imaging_B1_253

% load B6 253
imaging_B6_253

% load B3 253
imaging_B3_waves

%% performance arrow task 

clear
clc
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))


%%%%% performance B1 arrow
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

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/')
folders={'20240515', '20240517', '20240614', ...
    '20240619', '20240621', '20240626',...
    '20240710','20240712','20240731'};
acc_days=[];
binomial_res=[];
conf_matrix_days=[];
num_trials_B1=[];
for days=1:length(folders)-1 %if B1-> it is -1

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

    % get decoding accuracy block by block
    acc_folders=[];bino_pdf_folders=[];
    confusion_mat=[];
    for ii=1:length(online_idx)
        online_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [findfiles('mat',online_folderpath)'];
        if rem(length(files),9) == 0
            [acc,acc1,bino_pdf] = accuracy_online_data_9DOF(files);
        else
            [acc,acc1,bino_pdf] = accuracy_online_data(files);
        end
        %acc = acc(1:7,1:7);
        acc_folders(ii) = mean(diag(acc));
        bino_pdf_folders(ii) = bino_pdf.pval;
        confusion_mat(ii,:,:) = acc(1:7,1:7);
        %disp(length(files))
    end
    l = round(length(acc_folders)/2);
    acc_days(days) = median(acc_folders(l:end));
    binomial_res = [binomial_res bino_pdf_folders(l:end)];
    conf_matrix_days(days,:,:) = squeeze(nanmean(confusion_mat(l:end,:,:),1));
    num_trials_B1(days) = length(acc_folders(l:end));
end

b1_acc= acc_days;
[pfdr,pmask]=fdr(binomial_res,0.05);
sum(binomial_res<=pfdr)/length(binomial_res)
figure;imagesc(squeeze(nanmean(conf_matrix_days,1)))
caxis([0 1])
tmp=squeeze(nanmean(conf_matrix_days,1))
mean(diag(tmp))
mean(b1_acc)
b1_conf_matrix=tmp;


%%%%% B3 arrow performance
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
len_days = min(11,length(session_data));
num_targets=7;
acc_days=[];binomial_res=[];
conf_matrix_days=[];
num_trials_B3=[];
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
    batch_idx = [batch_idx batch_idx1];
    %online_idx=[online_idx batch_idx batch_idx1];
    %online_idx = [batch_idx batch_idx1];

    folders = session_data(days).folders(batch_idx);
    day_date = session_data(days).Day;    
    acc_folders=[];bino_pdf_folders=[];confusion_mat=[];
    for ii=1:length(folders)
        %folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [findfiles('mat',folderpath)'];

        if rem(length(files),9) == 0
            [acc,acc1,bino_pdf] = accuracy_online_data_9DOF(files);
        else
            [acc,acc1,bino_pdf] = accuracy_online_data(files);
        end
        acc_folders(ii) = mean(diag(acc));
        bino_pdf_folders(ii) = bino_pdf.pval;
        confusion_mat(ii,:,:) = acc;
    end
    acc_days(days) = nanmedian(acc_folders);
    binomial_res = [binomial_res bino_pdf_folders];
    conf_matrix_days(days,:,:) = squeeze(nanmean(confusion_mat(1:end,:,:),1));
    num_trials_B3(days) = length(acc_folders(1:end));
end


b3_acc= acc_days;
[pfdr,pmask]=fdr(binomial_res,0.05);
sum(binomial_res<=pfdr)/length(binomial_res)
figure;imagesc(squeeze(nanmean(conf_matrix_days,1)))
caxis([0 1])
tmp=squeeze(nanmean(conf_matrix_days,1))
mean(diag(tmp))
mean(b3_acc)
b3_conf_matrix = tmp;

%%%%% b6 arrow performance 
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


cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
folders = {'20250530','20250610','20250624','20250703','20250708','20250717',...
    '20250917','20250924','20251203','20251204','20251210','20260116'};
acc_days=[];
binomial_res=[];
conf_matrix_days=[];
num_trials_B6=[];
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

    % get decoding accuracy block by block
    acc_folders=[];bino_pdf_folders=[];confusion_mat=[];
    for ii=1:length(online_idx)
        online_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [findfiles('mat',online_folderpath)'];
        if rem(length(files),9) == 0
            [acc,acc1,bino_pdf] = accuracy_online_data_9DOF(files);
        else
            [acc,acc1,bino_pdf] = accuracy_online_data(files);
        end
        %acc = acc(1:7,1:7);
        acc_folders(ii) = mean(diag(acc));
        bino_pdf_folders(ii) = bino_pdf.pval;
        confusion_mat(ii,:,:) = acc;
        %disp(length(files))
    end
    l = round(length(acc_folders)/2);
    acc_days(days) = nanmedian(acc_folders(l:end));
    binomial_res = [binomial_res bino_pdf_folders(l:end)];
    conf_matrix_days(days,:,:) = squeeze(nanmean(confusion_mat(l:end,:,:),1));
    num_trials_B6(days) = length(acc_folders(l:end));
end

b6_acc= acc_days;
[pfdr,pmask]=fdr(binomial_res,0.05);
sum(binomial_res<=pfdr)/length(binomial_res)
figure;imagesc(squeeze(nanmean(conf_matrix_days,1)))
caxis([0 1])
tmp=squeeze(nanmean(conf_matrix_days,1))
mean(diag(tmp))
mean(b6_acc)
b6_conf_matrix  = tmp;


num_trials_B1 = num_trials_B1*21;
num_trials_B3 = num_trials_B3*21;
num_trials_B6 = num_trials_B6*21;

[mean(num_trials_B1) mean(num_trials_B3) mean(num_trials_B6)]
ab = sort(bootstrp(1000,@mean,num_trials_B1));
[ab(25) ab(975)]
ab = sort(bootstrp(1000,@mean,num_trials_B3));
[ab(25) ab(975)]
ab = sort(bootstrp(1000,@mean,num_trials_B6));
[ab(25) ab(975)]

cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves')
save arrow_decoding_results_waves b1_acc b3_acc b6_acc b1_conf_matrix...
    b3_conf_matrix b6_conf_matrix -v7.3


% plot results
res=[b1_acc b3_acc b6_acc];
figure;
hold on
idx = ones(size(b1_acc)) + 0.025*randn(size(b1_acc));
plot(idx,b1_acc,'.','MarkerSize',30,'Color',[.5 .5 .5 .5])
idx = ones(size(b3_acc)) + 0.025*randn(size(b3_acc));
plot(idx,b3_acc,'.','MarkerSize',30,'Color',[.75 .0 .25 .5])
idx = ones(size(b6_acc)) + 0.025*randn(size(b6_acc));
plot(idx,b6_acc,'.','MarkerSize',30,'Color',[.25 .0 .75 .5])
plot_beautify
ylim([0.5 1])
xlim([.75 1.25])
xticks ''
yticks(.5:.1:1)

%%% better plotting
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

ylim([0 1])
yticks([0:.1:1])
xlim([.5 1.5])
yticks([0:.1:1])
ylim([.5 1])
h=hline(1/7);
set(h,'LineWidth',1)
xticks ''
plot_beautify

mb = bootstrp(1000,@median,res);
mb=sort(mb);
[mb(25) median(res) mb(975)]


% confusion matrices
acc_online = b1_conf_matrix;
figure;
imagesc(acc_online*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_online,1)
    for k=1:size(acc_online,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'X+ve/Rt Thumb','Y-ve/Leg','X-ve/Lt. Thumb','Y+ve/Head','Z+ve/Tong','Z-ve/Lips',...
    'Origin/Both middle'})
yticklabels({'X+ve/Rt Thumb','Y-ve/Leg','X-ve/Lt. Thumb','Y+ve/Head','Z+ve/Tong','Z-ve/Lips',...
    'Origin/Both middle'})
title(['B1: Mean Acc ' num2str(100*mean(diag(acc_online)))])



% confusion matrices
acc_online = b1_conf_matrix;
figure;
imagesc(acc_online*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_online,1)
    for k=1:size(acc_online,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'X+ve/Rt Thumb','Y-ve/Leg','X-ve/Lt. Thumb','Y+ve/Head','Z+ve/Tong.','Z-ve/Lips',...
    'Origin/Both middle'})
yticklabels({'X+ve/Rt Thumb','Y-ve/Leg','X-ve/Lt. Thumb','Y+ve/Head','Z+ve/Tong.','Z-ve/Lips',...
    'Origin/Both middle'})
title(['B1: Mean Acc ' num2str(100*mean(diag(acc_online)))])

% confusion matrices
acc_online = b3_conf_matrix;
figure;
imagesc(acc_online*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_online,1)
    for k=1:size(acc_online,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'X+ve/Rt Thumb','Y-ve/Leg','X-ve/Lt. Thumb','Y+ve/Head','Z+ve/Tong.','Z-ve/Lips',...
    'Origin/Both middle'})
yticklabels({'X+ve/Rt Thumb','Y-ve/Leg','X-ve/Lt. Thumb','Y+ve/Head','Z+ve/Tong.','Z-ve/Lips',...
    'Origin/Both middle'})
title(['B3: Mean Acc ' num2str(100*mean(diag(acc_online)))])


% confusion matrices
acc_online = b6_conf_matrix;
figure;
imagesc(acc_online*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_online,1)
    for k=1:size(acc_online,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'X+ve/Rt Thumb','Y-ve/Leg','X-ve/Lt. Thumb','Y+ve/Head','Z+ve/Tong','Z-ve/Lips',...
    'Origin/Both middle'})
yticklabels({'X+ve/Rt Thumb','Y-ve/Leg','X-ve/Lt. Thumb','Y+ve/Head','Z+ve/Tong','Z-ve/Lips',...
    'Origin/Both middle'})
title(['B6: Mean Acc ' num2str(100*mean(diag(acc_online)))])


%% performance B3 hand OL, CL all


clc;clear;
close all

addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))

root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
cd(root_path)
addpath('/home/user/Documents/MATLAB/DrosteEffect-BrewerMap-5b84f95/')
load session_data_B3_Hand
addpath '/home/user/Documents/MATLAB'




acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
acc_batch_bin_days=[];
iterations=1;
plot_true=true;
acc_batch_days_overall=[];
condn_data_imagined = cell(1,12);
condn_data_online = cell(1,12);
condn_data_batch= cell(1,12);
mahab_full_imagined=[];
mahab_full_online=[];
mahab_full_batch=[];
num_trials=[];
for i=1:length(session_data) % 20230518 has the best hand data performance
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    if sum(folders_batch1)==0
        batch_idx = find(folders_batch==1);
    else
        batch_idx = find(folders_batch1==1);
    end



    disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);


    % save the data
    tmp={};
    for ii=1:12
        idx = find([condn_data.targetID]==ii);
        xx=[];
        for jj=1:length(idx)
            xx = [xx condn_data(idx(jj)).neural];
        end
        tmp{ii}=xx';
    end
    condn_data_bins = tmp;
    %condn_data = condn_data_bins;
    %filename = ['condn_data_Hand_B3_ImaginedTrials_Day' num2str(i)];
    %save(filename, 'condn_data_bins', '-v7.3')

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data_bins);
    z=linkage(squareform(Dimagined),'ward');
    figure;dendrogram(z)
    %Dimagined = triu(Dimagined);
    %Dimagined = Dimagined(Dimagined>0);
    Dimagined=squareform(Dimagined);
    mahab_full_imagined = [mahab_full_imagined Dimagined'];

    % get relative means and store
    %condn_data_bins = normalize_acrossDays(condn_data_bins);
    for ii=1:length(condn_data_bins)
        xx=condn_data_imagined{ii};
        yy=condn_data_bins{ii};
        xx=[xx;yy];
        condn_data_imagined{ii}=xx;
    end

    % get cross-val classification accuracy
    [acc_imagined,train_permutations,~,bino_pdf] = ...
        accuracy_imagined_data_Hand_B3(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_imagined,1)
            for k=1:size(acc_imagined,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_imagined(k,j),1)),'Color','k')
                end
            end
        end
        box on
        title(['OL Accuracy of ' num2str(100*mean(diag(acc_imagined)))])
        xticks(1:12)
        yticks(1:12)
        xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
    end
    acc_imagined_days(:,:,i) = (acc_imagined);
    % store binomial results
    n = round(median([bino_pdf(1:end).n]));
    succ = round(median([bino_pdf(1:end).succ]));
    pval = binopdf(succ,n,(1/12));
    binomial_res(i).Imagined = [pval];


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);

    % save the data
    tmp={};
    for ii=1:12
        idx = find([condn_data.targetID]==ii);
        xx=[];
        for jj=1:length(idx)
            xx = [xx condn_data(idx(jj)).neural];
        end
        tmp{ii}=xx';
    end
    condn_data_bins = tmp;
    %condn_data = condn_data_bins;
    %filename = ['condn_data_Hand_B3_OnlineTrials_Day' num2str(i)];
    %save(filename, 'condn_data_bins', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data_bins);
    z=linkage(squareform(Donline),'ward');
    figure;dendrogram(z)
    Donline = squareform(Donline);
    %Donline = triu(Donline);
    %Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online (Donline)'];

    % get relative means and store
    %condn_data_bins = normalize_acrossDays(condn_data_bins);
    for ii=1:length(condn_data_bins)
        xx=condn_data_online{ii};
        yy=condn_data_bins{ii};
        xx=[xx;yy];
        condn_data_online{ii}=xx;
    end

    % get the classification accuracy
    [acc_online,acc_online_bin,trial_len,bino_pdf] = accuracy_online_data_Hand(files,12);
    if plot_true
        figure;imagesc(acc_online*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_online,1)
            for k=1:size(acc_online,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_online(k,j),1)),'Color','k')
                end
            end
        end
        title(['CL 1 Accuracy of ' num2str(100*mean(diag(acc_online)))])
        box on
        xticks(1:12)
        yticks(1:12)
        xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
    end
    acc_online_days(:,:,i) = (acc_online);
    binomial_res(i).online = [bino_pdf.pval];


    %%%%%% classification accuracy for batch data
    % first get all the batch data and save
    folders = session_data(i).folders(batch_idx_overall);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);

    % save the data
    tmp={};
    for ii=1:12
        idx = find([condn_data.targetID]==ii);
        xx=[];
        for jj=1:length(idx)
            xx = [xx condn_data(idx(jj)).neural];
        end
        tmp{ii}=xx';
    end
    condn_data_bins = tmp;
    %condn_data = condn_data_bins;
    %filename = ['condn_data_Hand_B3_BatchTrials_Day' num2str(i)];
    %save(filename, 'condn_data_bins', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data_bins);
    z=linkage(squareform(Donline),'ward');
    figure;dendrogram(z)
    Donline = squareform(Donline);
    %Donline = Donline(Donline>0);
    mahab_full_batch = [mahab_full_batch Donline'];

    % get relative means and store
    %condn_data_bins = normalize_acrossDays(condn_data_bins);
    for ii=1:length(condn_data_bins)
        xx=condn_data_batch{ii};
        yy=condn_data_bins{ii};
        xx=[xx;yy];
        condn_data_batch{ii}=xx;
    end

    % for classification, get only the latest batch files
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    num_trials = [num_trials length(files)];

    % get the classification accuracy
    [acc_batch,acc_batch_bin,trial_len,bino_pdf,decodes_trial] = ...
        accuracy_online_data_Hand(files,12);
    
    if plot_true
        figure;imagesc(acc_batch*100)
        colormap(brewermap(128,'Blues'))
        clim([0 100])
        set(gcf,'color','w')
        % add text
        for j=1:size(acc_batch,1)
            for k=1:size(acc_batch,2)
                if j==k
                    text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','w')
                else
                    text(j-0.35,k,num2str(round(100*acc_batch(k,j),1)),'Color','k')
                end
            end
        end
        title(['CL 2 Accuracy of ' num2str(100*mean(diag(acc_batch)))])
        box on
        xticks(1:12)
        yticks(1:12)
        xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})

    end
    acc_batch_days(:,:,i) = (acc_batch);
    binomial_res(i).batch = [bino_pdf.pval];
    acc_batch_bin_days(:,:,i) = acc_batch_bin;
    close all

    % get the decodes per trial 
    %decoders_trial =  get_decodes_trial(files);

end

%save B3_Hand_Res -v7.3
% save B3_Hand_Res_ForPaper -v7.3

% plotting clustering
tmp = mean(mahab_full_batch(:,1:10),2);
z=linkage(tmp','ward');
figure;
dendrogram(z)

tmp = (mahab_full_imagined(:,8));
z=linkage(tmp','complete');
figure;
dendrogram(z)


%%%%%% (MAIN) MDS CODE WITH STATS ON PRESERVING REPRESENTATIONAL STRUCTURE
% ALONG WITH CORRELATING DECODING ACC WITH MEAN MAHAB DISTANCE PER DAY
dec_acc=[];
for i=1:size(acc_batch_days,3)
    a = squeeze(acc_batch_days(:,:,i));
    dec_acc(i)= mean(diag(a));
end

% ============================================================
% Closed-loop representational geometry analysis
% mahab_full_batch: 66 x nDays
% dec_acc: nDays x 1 or 1 x nDays
% ============================================================

CLvecs = mahab_full_batch;
dec_acc = dec_acc(:);

classLabels = {'Thumb','Index','Middle','Ring','Pinky','Power', ...
               'Pinch','Tripod','Wrist Add.','Wrist Abd.', ...
               'Wrist Flex','Wrist Ext.'};

nClasses = 12;
nDays = size(CLvecs,2);

assert(size(CLvecs,1) == nClasses*(nClasses-1)/2, ...
    'mahab_full_batch should be 66 x nDays');

assert(numel(dec_acc) == nDays, ...
    'dec_acc should have one value per day');

% ============================================================
% 1. Convert squareform vectors back to distance matrices
% ============================================================

D_CL = zeros(nClasses,nClasses,nDays);

for d = 1:nDays
    y = CLvecs(:,d)';
    D = squareform(y);

    D = (D + D')/2;
    D(1:nClasses+1:end) = 0;

    D_CL(:,:,d) = D;
end

% ============================================================
% 2. MDS of average closed-loop distance matrix
% ============================================================

Dmean_CL = mean(D_CL,3);

[Y_CL,eig_CL] = cmdscale(Dmean_CL);

posEig = eig_CL(eig_CL > 0);
varExplained = eig_CL ./ sum(posEig);

figure;
scatter3(Y_CL(:,1), Y_CL(:,2), Y_CL(:,3),180, 'filled');
hold on;

text(Y_CL(:,1), Y_CL(:,2),Y_CL(:,3), classLabels, ...
    'VerticalAlignment','bottom', ...
    'HorizontalAlignment','right', ...
    'FontSize',12);

xlabel(sprintf('MDS 1 %.1f%%',100*varExplained(1)));
ylabel(sprintf('MDS 2 %.1f%%',100*varExplained(2)));
zlabel(sprintf('MDS 3 %.1f%%',100*varExplained(3)));
title('Mean closed-loop representational geometry');
%axis equal;
grid on;
view(-90,90)

% Optional: hierarchical clustering of mean CL geometry

Zmean = linkage(squareform(Dmean_CL), 'average');

figure;
dendrogram(Zmean, 'Labels', classLabels);
ylabel('Mahalanobis distance');
title('Hierarchical clustering of mean CL geometry');

% Choose a cutoff based on dendrogram
cutoff = 440;   % adjust as needed
clusterID = cluster(Zmean, 'cutoff', cutoff, 'criterion', 'distance');

figure;
scatter(Y_CL(:,1), Y_CL(:,2), 180, clusterID, 'filled');
hold on;

text(Y_CL(:,1), Y_CL(:,2), classLabels, ...
    'VerticalAlignment','bottom', ...
    'HorizontalAlignment','right', ...
    'FontSize',12);

xlabel(sprintf('MDS 1 %.1f%%',100*varExplained(1)));
ylabel(sprintf('MDS 2 %.1f%%',100*varExplained(2)));
title('Mean CL MDS colored by automatic clusters');
axis equal;
grid on;
colorbar;

disp(table(classLabels(:), clusterID(:), ...
    'VariableNames', {'Class','Cluster'}));

% ============================================================
% 3. Geometry preservation across days
% Leave-one-day-out correlation:
% each day compared to mean geometry of the other days
% ============================================================

r_LOO_spearman = zeros(nDays,1);
r_LOO_pearson  = zeros(nDays,1);

for testDay = 1:nDays

    trainDays = setdiff(1:nDays,testDay);

    testVec = CLvecs(:,testDay);
    trainMeanVec = mean(CLvecs(:,trainDays),2);

    r_LOO_spearman(testDay) = corr(testVec, trainMeanVec, ...
        'type','Spearman', 'rows','complete');

    r_LOO_pearson(testDay) = corr(testVec, trainMeanVec, ...
        'type','Pearson', 'rows','complete');
end

fprintf('\n=== Closed-loop geometry preservation across days ===\n');
fprintf('LOO Spearman r = %.3f ± %.3f\n', ...
    mean(r_LOO_spearman), std(r_LOO_spearman));

fprintf('LOO Pearson r  = %.3f ± %.3f\n', ...
    mean(r_LOO_pearson), std(r_LOO_pearson));

% Nonparametric test vs zero
p_LOO_spearman = signrank(r_LOO_spearman, 0);
p_LOO_pearson  = signrank(r_LOO_pearson, 0);

fprintf('LOO Spearman signrank vs 0: p = %.5f\n', p_LOO_spearman);
fprintf('LOO Pearson signrank vs 0:  p = %.5f\n', p_LOO_pearson);

% Fisher-z t-test
z_LOO = atanh(r_LOO_spearman);
[h_z,p_z,ci_z,stats_z] = ttest(z_LOO,0);

fprintf('Fisher-z t-test: t(%d)=%.3f, p=%.5f\n', ...
    stats_z.df, stats_z.tstat, p_z);

% Plot LOO correlations

figure;
bar(r_LOO_spearman);
yline(0,'k--');
ylim([-1 1]);
xlabel('Held-out day');
ylabel('Spearman correlation');
title(sprintf('CL geometry stability across days: mean r = %.2f, p = %.4g', ...
    mean(r_LOO_spearman), p_LOO_spearman));
grid on;

figure;
boxplot(r_LOO_spearman,'Whisker',3)
ylim([-0.5 1]);
yline(0,'k--');
xlim([.85 1.15])
plot_beautify
xticks ''


% ================================
% 4: correlation for held out day spearman to average geometry from other
% days 
% ==================

% Closed-loop geometry consolidation analysis
% mahab_full_batch: 66 x nDays


% Compute held-out day Spearman correlations

r_LOO = zeros(nDays,1);
p_LOO = zeros(nDays,1);
for testDay = 1:nDays

    otherDays = setdiff(1:nDays, testDay);

    testVec = CLvecs(:,testDay);
    meanOtherVec = mean(CLvecs(:,otherDays), 2);

    [r_LOO(testDay), p_LOO(testDay)] = corr(testVec, meanOtherVec, ...
        'type', 'Spearman', ...
        'rows', 'complete');
end

dayNum = (1:nDays)';

T = table(dayNum, r_LOO, ...
    'VariableNames', {'Day','SpearmanR'});

% Regular Huber robust regression, no Fisher z-transform

mdl_huber = fitlm(T, 'SpearmanR ~ Day', ...
    'RobustOpts', 'huber');

disp(mdl_huber);
disp(mdl_huber.Coefficients);

beta_day = mdl_huber.Coefficients.Estimate("Day");
p_day    = mdl_huber.Coefficients.pValue("Day");

fprintf('\nHuber robust regression on raw Spearman r:\n');
fprintf('Beta day = %.4f, p = %.5f\n', beta_day, p_day);

% plot
figure;
scatter(dayNum, r_LOO, 120, 'filled');
hold on;

xx = linspace(1, nDays, 200)';
Tpred = table(xx, 'VariableNames', {'Day'});

rpred = predict(mdl_huber, Tpred);

plot(xx, rpred, 'k-', 'LineWidth', 2);
yline(0, 'k--');

xlabel('Closed-loop day');
ylabel('Spearman correlation with other-day mean geometry');

title(sprintf(['Closed-loop representational geometry consolidates across days\n' ...
               'Huber robust regression: \\beta = %.3f, p = %.4g'], ...
               beta_day, p_day));

ylim([-1 1]);
xlim([0.5 nDays+0.5]);
grid on;


% Test held-out correlations against zero

p_signrank = signrank(r_LOO, 0);

fprintf('\nHeld-out correlations vs zero:\n');
fprintf('Mean Spearman r = %.3f ± %.3f\n', mean(r_LOO), std(r_LOO));
fprintf('Median Spearman r = %.3f\n', median(r_LOO));
fprintf('Signrank p = %.5f\n', p_signrank);

% Box/point plot

figure;
boxchart(ones(nDays,1), r_LOO);
hold on;

scatter(ones(nDays,1), r_LOO, 90, 'filled', ...
    'jitter', 'on', ...
    'jitterAmount', 0.08);

yline(0, 'k--');

xlim([0.5 1.5]);
xticks(1);
xticklabels({'Held-out days'});
ylabel('Spearman correlation with other-day mean geometry');

title(sprintf('Held-out correlations > 0, signrank p = %.4g', p_signrank));

ylim([-1 1]);
grid on;


%%%%%%%%% end



% plotting confusion matrix of last day 
a=squeeze(acc_batch_days(:,:,end));
figure;imagesc(a)
colormap parula
caxis([0 1])
xticks(1:12)
yticks(1:12)
xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist Add.','Wrist Abd.','Wrist Flex','Wrist Ext.'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist Add.','Wrist Abd.','Wrist Flex','Wrist Ext.'})
plot_beautify

%%%%% plotting MDS
classLabels = {'Thumb','Index','Middle','Ring','Pinky','Power', ...
               'Pinch','Tripod','Wrist Add.','Wrist Abd.', ...
               'Wrist Flex','Wrist Ext.'};
D = mean(mahab_full_imagined,2);
D = squareform(D);
D = (D + D')/2;
D(1:size(D,1)+1:end) = 0;

% Classical MDS
[Y, eigvals] = cmdscale(D);

% Variance explained
posEig = eigvals(eigvals > 0);
varExplained = eigvals ./ sum(posEig);

% Plot first two MDS dimensions
figure;
scatter(Y(:,1), Y(:,2), 150, 'filled'); 
hold on;

text(Y(:,1), Y(:,2), classLabels, ...
    'VerticalAlignment','bottom', ...
    'HorizontalAlignment','right', ...
    'FontSize',12);

xlabel(sprintf('MDS 1 %.1f%%', 100*varExplained(1)));
ylabel(sprintf('MDS 2 %.1f%%', 100*varExplained(2)));
title('MDS of Class Mahalanobis Distance Matrix');
%axis equal;
grid on;

% Plot first three MDS dimensions
figure;
scatter3(Y(:,1), Y(:,2), Y(:,3), 150, 'filled'); 
hold on;

text(Y(:,1), Y(:,2), Y(:,3), classLabels, ...
    'VerticalAlignment','bottom', ...
    'HorizontalAlignment','right', ...
    'FontSize',12);

xlabel(sprintf('MDS 1 %.1f%%', 100*varExplained(1)));
ylabel(sprintf('MDS 2 %.1f%%', 100*varExplained(2)));
zlabel(sprintf('MDS 3 %.1f%%', 100*varExplained(3)));
title('MDS of Class Mahalanobis Distance Matrix');
%axis equal;
grid on;


%% MORE INDEPTH MDS 
% INPUT
% distVecs is 66 x 10
% rows = pairwise class distances from squareform(D)
% columns = days

classLabels = {'Thumb','Index','Middle','Ring','Pinky','Power', ...
               'Pinch','Tripod','Wrist Add.','Wrist Abd.', ...
               'Wrist Flex','Wrist Ext.'};

groups = [1 1 1 1 1 2 2 2 3 3 3 3];
groupNames = {'Digits','Grasps','Wrist'};
distVecs = mahab_full_imagined;

nClasses = 12;
nDays = size(distVecs,2);

% Clean and reconstruct distance matrices
Dall = zeros(nClasses, nClasses, nDays);

for d = 1:nDays
    y = distVecs(:,d);
    y = y(:)';                 % important: squareform expects row vector
    D = squareform(y);

    D = (D + D')/2;            % enforce symmetry
    D(1:nClasses+1:end) = 0;   % zero diagonal

    Dall(:,:,d) = D;
end

% Reference MDS from average distance matrix
Dmean = mean(Dall,3);

[Yref, eigvals] = cmdscale(Dmean);

% Use first 2 MDS dimensions
Yref2 = Yref(:,1:2);

posEig = eigvals(eigvals > 0);
varExplained = eigvals ./ sum(posEig);

fprintf('MDS1 = %.1f%%\n', 100*varExplained(1));
fprintf('MDS2 = %.1f%%\n', 100*varExplained(2));

% MDS for each day, aligned to reference using Procrustes
Ydays = zeros(nClasses, 2, nDays);

for d = 1:nDays
    D = Dall(:,:,d);

    [Y, eigvals_day] = cmdscale(D);
    Y2 = Y(:,1:2);

    % Align each day's MDS to the reference MDS
    % This removes arbitrary rotation/reflection/translation differences.
    [~, Yaligned] = procrustes(Yref2, Y2, ...
                               'Scaling', false, ...
                               'Reflection', false);

    Ydays(:,:,d) = Yaligned;
end

% Mean and covariance of each class location across days
Ymean = mean(Ydays,3);

figure;
hold on;

% Plot individual day points lightly
for d = 1:nDays
    scatter(Ydays(:,1,d), Ydays(:,2,d), 40, 'filled', ...
        'MarkerFaceAlpha', 0.25);
end

% Plot mean class locations
scatter(Ymean(:,1), Ymean(:,2), 160, 'filled');

text(Ymean(:,1), Ymean(:,2), classLabels, ...
    'VerticalAlignment','bottom', ...
    'HorizontalAlignment','right', ...
    'FontSize',12);

xlabel(sprintf('MDS 1 %.1f%%', 100*varExplained(1)));
ylabel(sprintf('MDS 2 %.1f%%', 100*varExplained(2)));
title('Day-aligned MDS with class variability');
axis equal;
grid on;


%% Plot 95% covariance ellipses for each class
figure;
hold on;

% Plot all daily class points
for d = 1:nDays
    scatter(Ydays(:,1,d), Ydays(:,2,d), 35, 'filled', ...
        'MarkerFaceAlpha', 0.25);
end

% Plot class means
scatter(Ymean(:,1), Ymean(:,2), 160, 'filled');

% Add labels
text(Ymean(:,1), Ymean(:,2), classLabels, ...
    'VerticalAlignment','bottom', ...
    'HorizontalAlignment','right', ...
    'FontSize',12);

% 95% ellipse scale for 2D Gaussian
chi2val = 5.991;  % chi2inv(0.95,2)

for c = 1:nClasses
    pts = squeeze(Ydays(c,:,:))';   % nDays x 2

    mu = mean(pts,1);
    C = cov(pts);

    % Avoid issues if covariance is nearly singular
    C = C + 1e-6 * eye(2);

    [V,L] = eig(C);

    theta = linspace(0, 2*pi, 200);
    circle = [cos(theta); sin(theta)];

    ellipse = V * sqrt(L * chi2val) * circle;
    ellipse = ellipse + mu';

    plot(ellipse(1,:), ellipse(2,:), 'LineWidth', 1.5);
end

xlabel(sprintf('MDS 1 %.1f%%', 100*varExplained(1)));
ylabel(sprintf('MDS 2 %.1f%%', 100*varExplained(2)));
title('MDS class locations with 95% day-to-day ellipses');
axis equal;
grid on;

%%

% plotting the accuracy
acc=[];acc_i=[];
for i=1:length(session_data)
    tmp = squeeze(acc_batch_days(:,:,i));
    acc(i) = 100*mean(diag(tmp));
    tmp = squeeze(acc_imagined_days(:,:,i));
    acc_i(i) = 100*mean(diag(tmp));
end
figure;hold on
plot(acc,'ok','MarkerSize',20)
%plot(acc,'LineWidth',2,'Color','k')
ylim([0 100])
xlabel('Days')
ylabel('Decoding Accuracy')
set(gcf,'Color','w')
box off
set(gca,'FontSize',12)
xlim([0 length(session_data)+1])
xticks(1:length(session_data))
yticks([0:10:100])
set(gca,'LineWidth',1)
hline(100/12,'--r')
hold on
% plot(acc_i,'ob','MarkerSize',20)
% plot(acc_i,'LineWidth',2,'Color','b')
% legend({,'','Closed loop','','Open loop'})

% sigmoid fit
x=1:10;
y=acc;
x = x(:);
y = y(:);
sigmoidModel = fittype( ...
    'd + (a-d)/(1 + exp(-b*(x-c)))', ...
    'independent', 'x', ...
    'coefficients', {'a','b','c','d'});
% Initial guesses
a0 = min(y);
d0 = max(y);
c0 = median(x);
b0 = 1;
opts = fitoptions(sigmoidModel);
opts.StartPoint = [a0 b0 c0 d0];
[f, gof] = fit(x, y, sigmoidModel, opts);
% Plot
%figure;
%plot(x, y, 'ko', 'MarkerFaceColor', 'k'); hold on;
xx = linspace(min(x), max(x), 300);
plot(xx, f(xx), 'k','LineWidth', 2);
%xlabel('x');
%ylabel('y');
%title('Sigmoid fit');

% plot accuaracy vs. mahab distances
x= median(mahab_full_online,1) - median(mahab_full_imagined,1);
y = acc;
figure;plot(x,y,'.k','MarkerSize',20)
[bhat p wh se ci t_stat]=robust_fit(x',y',1);

% plotting MDSCALE (from Day 1, OL and CL1)
ImaginedMvmt = {'Thumb','Index','Middle','Ring','Pinky','Power',...
    'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'};
figure;
[Y,~,disparities] = mdscale(Dimagined,2);
Ymds=Y;
hold on
cmap = turbo(length(ImaginedMvmt));
idx=[1:12];
for i=1:size(Y,1)
    plot(Y(i,1),Y(i,2),'.','Color',cmap(i,:),'MarkerSize',20)
    if sum(i==idx)==1
        text(Y(i,1),Y(i,2),ImaginedMvmt{i},'FontWeight','bold','FontSize',10);
    else
        text(Y(i,1),Y(i,2),ImaginedMvmt{i},'FontSize',10);
    end
end
xlim([-200 250])
ylim([-150 150])
set(gcf,'Color','w')
set(gca,'FontSize',12)
set(gca,'LineWidth',1)
title('Open loop representations')
xlabel('MDS Axis  1')
ylabel('MDS Axis  2')

% plottig dendrogram
Z=linkage(squareform(Dimagined),'complete');
figure;
dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')
xlabel('Movements')
ylabel('Mahalanobis distance')
title('Open Loop')

% plotting matrix
figure;imagesc(Dimagined)
box on
xticks(1:12)
yticks(1:12)
xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
    'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
    'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
set(gcf,'Color','w')
colorbar
set(gca,'FontSize',12)


% combining wrist actions into one class
acc_batch_new = zeros(10);
acc_batch_new(1:8,1:8) = acc_batch(1:8,1:8);
tmp = mean(acc_batch(9:10,1:8),1);
acc_batch_new(9,1:8) = tmp;
acc_batch_new(9,9) = 1-sum(tmp);
tmp = mean(acc_batch(11:12,1:10),1);
tmp=[tmp(1:8) sum(tmp(9:10))];
acc_batch_new(10,1:9)=tmp;
acc_batch_new(10,10) = 1-sum(tmp);
figure;imagesc(acc_batch_new*100)
colormap bone
clim([0 100])
set(gcf,'color','w')
set(gca,'FontSize',14);
title(['Accuracy of ' num2str(round(100*mean(diag(acc_batch_new)))) '%'])
xticks(1:10)
xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
    'Pinch','Tripod','Wrist Add/Abd','Wrist Flex/Extend'})
yticks(1:10)
yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
    'Pinch','Tripod','Wrist Add/Abd','Wrist Flex/Extend'})

% plotting bin level accuracies
for i=1:size(acc_batch_bin_days,3)
    figure;
    tmp=squeeze(acc_batch_bin_days(:,:,i));
    imagesc(tmp);
    title([num2str(mean(diag(tmp))) ' day '   num2str(i)])
    colormap bone
end

% plotting the accuracy over days
acc_batch_days_overall = acc_batch_days_overall(:,:,[1:3 5:end]);
acc_batch_days = acc_batch_days(:,[1:3 5:end]);
figure;
plot(1:size(acc_batch_days,2),mean(acc_batch_days,1),'k','LineWidth',1);
xticks(1:6)
xlabel('Days')
ylabel('Trial Level Accuracy')
title('12 Hand Actions CL Decoding')
ylim([0 1])
yticks([0:0.1:1])
hline(1/12,'r')
set(gca,'FontSize',12)
set(gcf,'Color','w')
box off
xlim([0.5 6.5])

% plotting the confusion matrix
tmp1=squeeze(nanmean(acc_batch_days_overall(:,:,end),3));
figure;imagesc(tmp1*100)
colormap bone
xticks(1:12)
yticks(1:12)
xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
    'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
    'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
set(gcf,'Color','w')
set(gca,'FontSize',10)
title(['Across day Accuracy of ' num2str(100*mean(diag(tmp1))) '%'])
clim([0 100])


%% performance B3 hand TRIALS ACROSS DAYS (ONLY CLOSED LOOP)


addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))

root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
cd(root_path)
addpath('/home/user/Documents/MATLAB/DrosteEffect-BrewerMap-5b84f95/')
load session_data_B3_Hand
addpath '/home/user/Documents/MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
acc_batch_bin_days=[];
iterations=1;
plot_true=false;
acc_batch_days_overall=[];
condn_data_imagined = cell(1,12);
condn_data_online = cell(1,12);
condn_data_batch= cell(1,12);
mahab_full_imagined=[];
mahab_full_online=[];
mahab_full_batch=[];
num_trials=[];
acc_CL_bins={};kk=1;

for i=1:length(session_data) % 20230518 has the best hand data performance
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    if sum(folders_batch1)==0
        batch_idx = find(folders_batch==1);
    else
        batch_idx = find(folders_batch1==1);
    end



    disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

   

    %%%%%% get classification accuracy for online data

    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        files=[];
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
        [acc_online,acc_online_bin,trial_len,bino_pdf,decodes_trial] = ...
            accuracy_online_data_Hand(files,12);
        acc_CL_bins(kk).CL = 'CL1';
        acc_CL_bins(kk).perf = mean(diag(acc_online_bin));
        acc_CL_bins(kk).perf_trial = mean(diag(acc_online));
        acc_CL_bins(kk).day = day_date;
        acc_CL_bins(kk).foldername = folders{ii};
        kk=kk+1;
    end
   
      
    acc_online_days(:,:,i) = (acc_online);
    binomial_res(i).online = [bino_pdf.pval];


    %%%%%% classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    
    for ii=1:length(folders)
        files=[];
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
        [acc_batch,acc_batch_bin,trial_len,bino_pdf,decodes_trial] = ...
            accuracy_online_data_Hand(files,12);
        acc_CL_bins(kk).CL = 'CL2';
        acc_CL_bins(kk).perf = mean(diag(acc_batch_bin));
        acc_CL_bins(kk).perf_trial = mean(diag(acc_batch));
        acc_CL_bins(kk).day = day_date;
        acc_CL_bins(kk).foldername = folders{ii};
        kk=kk+1;
    end
    num_trials = [num_trials length(files)];

    % get the classification accuracy
    
    
    
    acc_batch_days(:,:,i) = (acc_batch);
    binomial_res(i).batch = [bino_pdf.pval];
    acc_batch_bin_days(:,:,i) = acc_batch_bin;

    

    
    
    close all

    % get the decodes per trial 
    %decoders_trial =  get_decodes_trial(files);

end

% get CL2 performance across days
day_labels = unique({acc_CL_bins(1:end).day});
acc_cl2=[];
figure;
hold on
cmap = parula(10);
for i=1:length(day_labels)
    idx = (strcmp({acc_CL_bins.day}, day_labels{i})) .* ...
        (strcmp({acc_CL_bins.CL},'CL2'));
    tmp = [acc_CL_bins(logical(idx)).perf_trial];
    tmp=tmp(:);
    acc_cl2(i) = mean(tmp);
    idx = i+randn(size(tmp))*0.1;
    scatter(idx,100*tmp,50,'filled','Color',cmap(i,:),'LineWidth',1)
end
xticks(1:10)
xlim([0.5 10.5])
ylim([0 100])

% sigmoid fit
x=1:10;
y=acc_cl2*100;
x = x(:);
y = y(:);
sigmoidModel = fittype( ...
    'd + (a-d)/(1 + exp(-b*(x-c)))', ...
    'independent', 'x', ...
    'coefficients', {'a','b','c','d'});
% Initial guesses
a0 = min(y);
d0 = max(y);
c0 = median(x);
b0 = 1;
opts = fitoptions(sigmoidModel);
opts.StartPoint = [a0 b0 c0 d0];
[f, gof] = fit(x, y, sigmoidModel, opts);
% Plot
%figure;
%plot(x, y, 'ko', 'MarkerFaceColor', 'k'); hold on;
xx = linspace(min(x), max(x), 300);
plot(xx, f(xx), 'k','LineWidth', 2);


% plotting the accuracy
acc=[];acc_i=[];
for i=1:length(session_data)
    tmp = squeeze(acc_batch_days(:,:,i));
    acc(i) = 100*mean(diag(tmp));
    %tmp = squeeze(acc_imagined_days(:,:,i));
    %acc_i(i) = 100*mean(diag(tmp));
end
figure;hold on
plot(acc,'ok','MarkerSize',20)
plot(acc,'LineWidth',2,'Color','k')
ylim([0 100])
xlabel('Days')
ylabel('Decoding Accuracy')
set(gcf,'Color','w')
box off
set(gca,'FontSize',12)
xlim([0 length(session_data)+1])
xticks(1:length(session_data))
yticks([0:10:100])
set(gca,'LineWidth',1)
hline(100/12,'--r')
%hold on
%plot(acc_i,'ob','MarkerSize',20)
%plot(acc_i,'LineWidth',2,'Color','b')
%legend({,'','Closed loop','','Open loop'})

% sigmoid fit
x=1:10;
y=acc*100;
x = x(:);
y = y(:);
sigmoidModel = fittype( ...
    'd + (a-d)/(1 + exp(-b*(x-c)))', ...
    'independent', 'x', ...
    'coefficients', {'a','b','c','d'});
% Initial guesses
a0 = min(y);
d0 = max(y);
c0 = median(x);
b0 = 1;
opts = fitoptions(sigmoidModel);
opts.StartPoint = [a0 b0 c0 d0];
[f, gof] = fit(x, y, sigmoidModel, opts);
% Plot
%figure;
%plot(x, y, 'ko', 'MarkerFaceColor', 'k'); hold on;
xx = linspace(min(x), max(x), 300);
plot(xx, f(xx), 'LineWidth', 2);
%xlabel('x');
%ylabel('y');
%title('Sigmoid fit');




%% performance b1 b3 b6 arrow
% arrow




