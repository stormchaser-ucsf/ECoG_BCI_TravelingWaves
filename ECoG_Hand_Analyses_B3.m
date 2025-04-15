% main code for the hand project
%
% this includes the data recently collected with multipe sequences of hand
% movements

% overall, the methods to do are:
% covariance matrix and reimann classifiers of the hand actions
% covariance matrix and then using a GRU for low-D representation
% maybe a variational autoencoder for classification? Time-series
% traveling waves and seeing differences
% travling waves with a transformer




%%
disp('test')

%% SESSION DATA FOR HAND EXPERIMENTS B3

clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
cd(root_path)

%day1
session_data(1).Day = '20230510';
session_data(1).folders = {'114718','115447','120026','120552','121111','121639',...
    '122957','123819','124556','125329','130014',...
    '130904'};
session_data(1).folder_type={'I','I','I','I','I','I','O','O','O','O',...
    'O','B'};
session_data(1).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am'};

%day2
session_data(2).Day = '20230511';
session_data(2).folders = {'113750','114133','114535','115215','115650','120107',...
    '120841','121228','121645','122024',...
    '122813','123125','123502'};
session_data(2).folder_type={'I','I','I','I','I','I','O','O','O','O',...
    'B','B','B'};
session_data(2).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am','am'};

%day3
session_data(3).Day = '20230517';
session_data(3).folders = {'113943','114508','115054','115418','120017','120336',...
    '121248','121716','122111','122514',...
    '123247','123646','124013'};
session_data(3).folder_type={'I','I','I','I','I','I','O','O','O','O',...
    'B','B','B'};
session_data(3).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am','am'};


%day4
session_data(4).Day = '20230518';
session_data(4).folders = {'114942','115609','120009','120434','120825','121322',...
    '121645',...
    '122444','122837','123147','123506'...
    '124013','124254','124552'};
session_data(4).folder_type={'I','I','I','I','I','I','I','O','O','O','O'...
    'B','B','B'};
session_data(4).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am','am','am'};

%day5
session_data(5).Day = '20230523';
session_data(5).folders = {'132840','133418','133923','134303','135022','135355',...
    '140405','140709','141222','141527','142057','142936'};
session_data(5).folder_type={'I','I','I','I','I','I','O','O','O','O'...
    'B','B'};
session_data(5).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am'};

%day6
session_data(6).Day = '20230915';
session_data(6).folders = {'115710','120650','121145','121513','122018','122342',...
    '124125','124747',...
    '125417','125734','130403',...
    };
session_data(6).folder_type={'I','I','I','I','I','I','O','O',...
    'B','B','B'};
session_data(6).AM_PM = {'am','am','am','am','am','am',...
    'am','am','am','am','am'};


% % day 7
% session_data(7).Day = '20230920';
% session_data(7).folders = {'114228','114819','115158','115515','115941','120315',...
%     '121038','121409',...
%     '122026','122340',...
%     '122846','123144','123521','123826',...
%     };
% session_data(7).folder_type={'I','I','I','I','I','I','O','O',...
%     'B','B',...
%     'B1','B1','B1','B1'};
% session_data(7).AM_PM = {'am','am','am','am','am','am',...
%     'am','am','am','am','am','am','am','am'};

% day 8
session_data(7).Day = '20230922';
session_data(7).folders = {'120223','120932','121508','121835','122306','122633',...
    '124030','124610','125218',...
    '130158','130706','131319',...
    '131960'};
session_data(7).folder_type={'I','I','I','I','I','I','O','O','O',...
    'B','B','B','B1'};
session_data(7).AM_PM = {'am','am','am','am','am','am',...
    'am','am','am','am','am','am',...
    'am'};

%day9
session_data(8).Day = '20230929';
session_data(8).folders = {'114451','115446','115832','120436','120802','121150',...
    '122659','123018','123319',...
    '123911','124247','124513',...
    '124951','125216','125455'};
session_data(8).folder_type={'I','I','I','I','I','I','O','O','O',...
    'B','B','B','B1','B1','B1'};
session_data(8).AM_PM = {'am','am','am','am','am','am',...
    'am','am','am','am','am','am',...
    'am','am','am'};

% day 10 -> this miming plus movement data
session_data(9).Day = '20231006';
session_data(9).folders = {'114305','114835','115203','115533','115852','120227',...
    '120914','121221','121448',...
    '121955','122220','122451',...
    '122856','123132','123348',...
    };
session_data(9).folder_type={'I','I','I','I','I','I','O','O','O',...
    'B','B','B','B1','B1','B1'};
session_data(9).AM_PM = {'am','am','am','am','am','am',...
    'am','am','am','am','am','am',...
    'am','am','am'};


% day 11 -> this miming plus movement data
session_data(10).Day = '20231011';
session_data(10).folders = {'143224','143756','144132','145307','145629','150027',...
    '150717','151014','151304',...
    '151821','152025','152237',...
    '152647','152902','153129',...
    '153537','153736','153948'};
session_data(10).folder_type={'I','I','I','I','I','I','O','O','O',...
    'B','B','B','B','B','B1','B1','B1','B1'};
session_data(10).AM_PM = {'am','am','am','am','am','am',...
    'am','am','am','am','am','am','am',...
    'am','am','am','am','am'};

%save session_data_B3_Hand session_data

% for accuracy -> take B1 ie CL3 or more...
% for neural features, consider all CL2,3 as CL2.

%% PERFORMANCE IMAGINED - ONLINE- BATCH FOR B3 HAND EXPERIMENTS

clc;clear
close all
clc;clear;

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3_Hand
addpath 'C:\Users\nikic\Documents\MATLAB'
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
    filename = ['condn_data_Hand_B3_ImaginedTrials_Day' num2str(i)];
    save(filename, 'condn_data_bins', '-v7.3')

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data_bins);
    z=linkage(squareform(Dimagined),'ward');
    figure;dendrogram(z)
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

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
    filename = ['condn_data_Hand_B3_OnlineTrials_Day' num2str(i)];
    save(filename, 'condn_data_bins', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data_bins);
    z=linkage(squareform(Donline),'ward');
    figure;dendrogram(z)
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];

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
    filename = ['condn_data_Hand_B3_BatchTrials_Day' num2str(i)];
    save(filename, 'condn_data_bins', '-v7.3')

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
    num_trials = [num_trials length(files)]

    % get the classification accuracy
    [acc_batch,acc_batch_bin,trial_len,bino_pdf] = accuracy_online_data_Hand(files,12);
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

end

%save B3_Hand_Res -v7.3

% plotting the accuracy
acc=[];
for i=1:length(session_data)
    tmp = squeeze(acc_batch_days(:,:,i));
    acc(i) = mean(diag(tmp));
end
figure;hold on
plot(acc,'ok','MarkerSize',20)
plot(acc,'LineWidth',2,'Color','k')
ylim([0 1])
xlabel('Days')
ylabel('Decoding Accuracy (X100%)')
set(gcf,'Color','w')
box off
set(gca,'FontSize',12)
xlim([0 length(session_data)+1])
xticks(1:length(session_data))
yticks([0:.1:1])
set(gca,'LineWidth',1)

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

%% TESTING THE CONDITIONAL PCA APPROACH FOR HAND DATA CLASSIFICATION


clc;clear
close all
clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3_Hand
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=1;
plot_true=true;
acc_batch_days_overall=[];


% load all the imagined and online data
i=3;
folders_imag =  strcmp(session_data(i).folder_type,'I');
folders_online = strcmp(session_data(i).folder_type,'O');
folders_batch = strcmp(session_data(i).folder_type,'B');

imag_idx = find(folders_imag==1);
online_idx = find(folders_online==1);
batch_idx = find(folders_batch==1);


%disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

%%%%%% Imagined data
folders = session_data(i).folders(imag_idx);
day_date = session_data(i).Day;
files=[];
for ii=1:length(folders)
    folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
    %cd(folderpath)
    files = [files;findfiles('',folderpath)'];
end

%%%%%% Online data
folders = session_data(i).folders(online_idx);
day_date = session_data(i).Day;
for ii=1:length(folders)
    folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %cd(folderpath)
    files = [files;findfiles('',folderpath)'];
end

% load the data using PCA
load('ECOG_Grid_8596_000067_B3.mat')
[condn_data,coeff,score,latent] = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);

% build the model, first 150 PCs, calibration on held out data
[acc_imagined,train_permutations] = ...
    accuracy_imagined_data_Hand_B3_PCA(condn_data, 5,coeff,100);
[acc_imagined,train_permutations] = ...
    accuracy_imagined_data_Hand_B3(condn_data, 5);
acc_imagined = squeeze(nanmean(acc_imagined,1));
figure;imagesc(acc_imagined)
mean(diag(acc_imagined))

% build the model, first 150 PCs, calibration on all data
[acc_imagined,train_permutations] = ...
    accuracy_imagined_data_Hand_B3_PCA(condn_data, 5,coeff,100);
acc_imagined = squeeze(nanmean(acc_imagined,1));
figure;imagesc(acc_imagined)
mean(diag(acc_imagined))


% test the model's performance on batch data


%%%%%% classification accuracy for batch data
folders = session_data(i).folders(batch_idx);
day_date = session_data(i).Day;
files=[];
for ii=1:length(folders)
    folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %cd(folderpath)
    files = [files;findfiles('',folderpath)'];
end

% get the classification accuracy
acc_batch = accuracy_online_data_Hand(files,12);
if plot_true
    figure;imagesc(acc_batch)
    colormap bone
    clim([0 1])
    set(gcf,'color','w')
    title(['Accuracy of ' num2str(100*mean(diag(acc_batch)))])
    xticks(1:12)
    yticks(1:12)
    xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
        'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
    yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
        'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})

end
acc_batch_days(:,i) = diag(acc_batch);
acc_batch_days_overall(:,:,i)=acc_batch;



%% STEP 1: LOOK AT COVARIANCE MATRIX

clc;clear

% filter design
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [0.5,4]; % low pass
Params.FilterBank(end+1).fpass = [4,8]; % theta
Params.FilterBank(end+1).fpass = [8,13]; % alpha
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2
Params.FilterBank(end+1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [20]; % raw

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end


%load a block of data and filter it, with markers.
folderpath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220610\20220610\HandImagined'
D=dir(folderpath);
foldernames={};
for j=3:length(D)
    foldernames = cat(2,foldernames,D(j).name);
end

files=[];
for i=1:length(foldernames)
    filepath = fullfile(folderpath,foldernames{i},'BCI_Fixed');
    files= [files; findfiles('',filepath)'];
end

delta=[];
theta=[];
alpha=[];
beta=[];
hg=[];
raw=[];
trial_len=[];
state1_len=[];
for i=1:length(files)
    load(files{i})
    if TrialData.TargetID==1
        raw_data = cell2mat(TrialData.BroadbandData');
        trial_len=[trial_len;size(raw_data,1)];
        raw=[raw;raw_data];
        idx=find(TrialData.TaskState==1);
        tmp=cell2mat(TrialData.BroadbandData(idx)');
        state1_len=[state1_len size(tmp,1)];
    end
end
trial_len_total=cumsum(trial_len);

% extracting band specific information
delta = filter(Params.FilterBank(1).b,...
    Params.FilterBank(1).a,...
    raw);
delta=abs(hilbert(delta));

theta = filter(Params.FilterBank(2).b,...
    Params.FilterBank(2).a,...
    raw);

alpha = filter(Params.FilterBank(3).b,...
    Params.FilterBank(3).a,...
    raw);

beta1 = filter(Params.FilterBank(4).b,...
    Params.FilterBank(4).a,...
    raw);
beta2 = filter(Params.FilterBank(5).b,...
    Params.FilterBank(5).a,...
    raw);
%beta1=(beta1.^2);
%beta2=(beta2.^2);
%beta = log10((beta1+beta2)/2);
beta = (abs(hilbert(beta1)) + abs(hilbert(beta2)))/2;

% hg filter bank approach -> square samples, log 10 and then average across
% bands
hg_bank=[];
for i=6:length(Params.FilterBank)-1
    tmp = filter(Params.FilterBank(i).b,...
        Params.FilterBank(i).a,...
        raw);
    %tmp=tmp.^2;
    tmp=abs(hilbert(tmp));
    hg_bank = cat(3,hg_bank,tmp);
end
%hg = log10(squeeze(mean(hg_bank,3)));
hg = (squeeze(mean(hg_bank,3)));

% lpf the raw
raw = filter(Params.FilterBank(end).b,...
    Params.FilterBank(end).a,...
    raw);
for j=1:size(raw,2)
    raw(:,j) = smooth(raw(:,j),100);
end

% now going and referencing each trial to state 1 data
raw_ep={};
delta_ep={};
beta_ep={};
hg_ep={};
trial_len_total=[0 ;trial_len_total];
for i=1:length(trial_len_total)-1
    % raw
    tmp_data = raw(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    raw_ep = cat(2,raw_ep,tmp_data);

    %delta
    tmp_data = delta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    delta_ep = cat(2,delta_ep,tmp_data);

    %beta
    tmp_data = beta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    beta_ep = cat(2,beta_ep,tmp_data);

    %hg
    tmp_data = hg(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    hg_ep = cat(2,hg_ep,tmp_data);
end


figure;
temp=cell2mat(hg_ep');
plot(temp(:,3));
axis tight
vline(trial_len_total,'r')


%
% figure;
% subplot(4,1,1)
% plot(raw(:,3))
% vline(trial_len,'r')
% title('raw')
% axis tight
%
% subplot(4,1,2)
% plot(abs(hilbert(delta(:,3))))
% vline(trial_len,'r')
% title('delta')
% axis tight
%
% subplot(4,1,3)
% plot(beta(:,3))
% vline(trial_len,'r')
% title('beta')
% axis tight
%
% subplot(4,1,4)
% plot(hg(:,4))
% vline(trial_len,'r')
% title('hg')
% axis tight
%
% sgtitle('Target 1, Ch3')
% set(gcf,'Color','w')

%hg erps - take the frst 7800 time points
hg_data=[];
for i=1:length(hg_ep)
    tmp=hg_ep{i};
    hg_data = cat(3,hg_data,tmp(1:7800,:));
end

figure;
subplot(2,1,1)
ch=106;
plot(squeeze(hg_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(hg_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Hg Ch 106 Left Leg')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
ylim([-5 10])

%delta erps - take the frst 7800 time points
delta_data=[];
for i=1:length(delta_ep)
    tmp=delta_ep{i};
    %     for j=1:size(tmp,2)
    %         tmp(:,j)=smooth(tmp(:,j),100);
    %     end
    delta_data = cat(3,delta_data,tmp(1:7800,:));
end

%figure;
subplot(2,1,2)
ch=106;
plot(squeeze(delta_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(delta_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Raw Ch 106 Left Leg')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
ylim([-5 5])

%
% ch=106;
% tmp=squeeze(delta_data(:,ch,:));
% for i=1:size(tmp,2)
%     figure;plot(tmp(:,i))
%     vline(TrialData.Params.InstructedDelayTime*1e3,'r')
%     vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
%     vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
%         +  TrialData.Params.CueTime*1e3, 'k')
%     hline(0)
% end
%  ylim([-3 3])


% plotting covariance of raw
tmp=zscore(raw);
figure;imagesc(cov(raw))
[c,s,l]=pca((raw));
chmap=TrialData.Params.ChMap;
tmp1=c(:,1);
figure;imagesc(tmp1(chmap))
figure;
stem(cumsum(l)./sum(l))




%% NEW HAND DATA MULTI CYCLIC
% extract single trials
% get the time-freq features in raw and in hG
% train a bi-GRU


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20220608','20220610','20220622','20220624'};
cd(root_path)

imagined_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'HandImagined')
    D=dir(folderpath);

    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        if ~exist(filepath)
            filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        end
        tmp=dir(filepath);
        imagined_files = [imagined_files;findfiles('',filepath)'];
    end
end


% load the data for the imagined files, if they belong to right thumb,
% index, middle, ring, pinky, pinch, tripod, power
D1i={};
D2i={};
D3i={};
D4i={};
D5i={};
D6i={};
D7i={};
D8i={};
D9i={};
D10i={};
D11i={};
D12i={};
D13i={};
D14i={};
for i=1:length(imagined_files)
    disp(i/length(imagined_files)*100)
    try
        load(imagined_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' files{j}]);
    end

    if file_loaded
        action = TrialData.TargetID;
        %disp(action)

        % find the bins when state 3 happened and then extract each
        % individual cycle (2.6s length) as a trial

        % get times for state 3 from the sample rate of screen refresh
        time  = TrialData.Time;
        time = time - time(1);
        idx = find(TrialData.TaskState==3) ;
        task_time = time(idx);

        % get the kinematics and extract times in state 3 when trials
        % started and ended
        kin = TrialData.CursorState;
        kin=kin(1,idx);
        kind = [0 diff(kin)];
        aa=find(kind==0);
        kin_st=[];
        kin_stp=[];
        for j=1:length(aa)-1
            if (aa(j+1)-aa(j))>1
                kin_st = [kin_st aa(j)];
                kin_stp = [kin_stp aa(j+1)-1];
            end
        end

        %getting start and stop times
        start_time = task_time(kin_st);
        stp_time = task_time(kin_stp);


        % get corresponding neural times indices
        %         neural_time  = TrialData.NeuralTime;
        %         neural_time = neural_time-neural_time(1);
        %         neural_st=[];
        %         neural_stp=[];
        %         st_time_neural=[];
        %         stp_time_neural=[];
        %         for j=1:length(start_time)
        %             [aa bb]=min(abs(neural_time-start_time(j)));
        %             neural_st = [neural_st; bb];
        %             st_time_neural = [st_time_neural;neural_time(bb)];
        %             [aa bb]=min(abs(neural_time-stp_time(j)));
        %             neural_stp = [neural_stp; bb-1];
        %             stp_time_neural = [stp_time_neural;neural_time(bb)];
        %         end

        % get the broadband data for each trial
        raw_data=cell2mat(TrialData.BroadbandData');

        % extract the broadband data (Fs-1KhZ) based on rough estimate of
        % the start and stop times from the kinematic data
        start_time_neural = round(start_time*1e3);
        stop_time_neural = round(stp_time*1e3);
        data_seg={};
        for j=1:length(start_time_neural)
            tmp = (raw_data(start_time_neural(j):stop_time_neural(j),:));
            tmp=tmp(1:round(size(tmp,1)/2),:);
            % pca step
            %m=mean(tmp);
            %[c,s,l]=pca(tmp,'centered','off');
            %tmp = (s(:,1)*c(:,1)')+m;
            data_seg = cat(2,data_seg,tmp);
        end

        if action==1
            D1i = cat(2,D1i,data_seg);
        elseif action==2
            D2i = cat(2,D2i,data_seg);
        elseif action==3
            D3i = cat(2,D3i,data_seg);
        elseif action==4
            D4i = cat(2,D4i,data_seg);
        elseif action==5
            D5i = cat(2,D5i,data_seg);
        elseif action==6
            D6i = cat(2,D6i,data_seg);
        elseif action==7
            D7i = cat(2,D7i,data_seg);
        elseif action==8
            D8i = cat(2,D8i,data_seg);
        elseif action==9
            D9i = cat(2,D9i,data_seg);
        elseif action==10
            D10i = cat(2,D10i,data_seg);
        elseif action==11
            D11i = cat(2,D11i,data_seg);
        elseif action==12
            D12i = cat(2,D12i,data_seg);
        elseif action==13
            D13i = cat(2,D13i,data_seg);
        elseif action==14
            D14i = cat(2,D14i,data_seg);
        end
    end
end

%% TRAVELING WAVES EXAMPLE

clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3_Hand
addpath 'C:\Users\nikic\Documents\MATLAB'

load('ECOG_Grid_8596_000067_B3.mat')

% imagined
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20231011\HandImagined\143756\Imagined\Data0001.mat')

%online
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20231011\HandOnline\153948\BCI_Fixed\Data0004.mat')

% filter the data in mu band range and then take a look at dynamics
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',12,'HalfPowerFrequency2',16, ...
    'SampleRate',1e3);
fvtool(bpFilt)


% get raw data
idx = find(TrialData.TaskState==1);
feat = TrialData.BroadbandData;
data  = cell2mat(feat(idx)');

% filter and plot the movie
filt_data = filtfilt(bpFilt,data);

figure;
for i=1:2:size(data,1)
    tmp = filt_data(i,:);
    imagesc(tmp(ecog_grid));
    title(num2str(i))
    colormap turbo
    pause(0.05)
end



%% loading data, filtering and extracting epochs for use in a CNN AE
% MAIN


clc;clear
close all

if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    load session_data_B3_Hand
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers'))

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves/helpers'))

end

xdata={};
ydata={};
labels=[];
labels_batch=[];
days=[];
mvmt_labels=[];
trial_number=[];
data={};

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',200);



hg_alpha_switch=false; %1 means get hG, 0 means get alpha dynamics

for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    %batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    batch_idx1 = find(folders_batch1==1);
    %     if sum(folders_batch1)==0
    %         batch_idx = find(folders_batch==1);
    %     else
    %         batch_idx = find(folders_batch1==1);
    %     end
    %     %batch_idx = [online_idx batch_idx];

    online_idx=[online_idx batch_idx];
    %batch_idx = [online_idx batch_idx_overall];


    %%%%%% get imagined data files
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx,trial_idx] = get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata);

    end

    labels = [labels; zeros(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];
    mvmt_labels= [mvmt_labels;trial_idx];

    %%%%%% getting online files now
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx] = get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata);

    end

    labels = [labels; ones(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];


    %%%%%% getting batch udpated (CL2) files now
    folders = session_data(i).folders(batch_idx1);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx] = get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata);

    end

    labels = [labels; ones(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;ones(idx,1)];

end

%save alpha_dynamics_200Hz_AllDays_zscore xdata ydata labels labels_batch days -v7.3
save alpha_dynamics_hG_200Hz_AllDays_DaysLabeled xdata ydata labels labels_batch days -v7.3



%% loading data, filtering and extracting epochs for use in a CNN AE, TRIAL FORMAT
% MAIN


clc;clear
close all

if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    load session_data_B3_Hand
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers'))

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves/helpers'))

end

xdata={};
ydata={};
labels=[];
labels_batch=[];
days=[];
mvmt_labels=[];
trial_number=[];
data_trial={};

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',200);

d2=d1;

hg_alpha_switch=false; %1 means get hG, 0 means get alpha dynamics

for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    %batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    batch_idx1 = find(folders_batch1==1);
    %     if sum(folders_batch1)==0
    %         batch_idx = find(folders_batch==1);
    %     else
    %         batch_idx = find(folders_batch1==1);
    %     end
    %     %batch_idx = [online_idx batch_idx];

    online_idx=[online_idx batch_idx];
    %batch_idx = [online_idx batch_idx_overall];


    %%%%%% get imagined data files
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    % trial level sort
    [data_trial] = get_spatiotemp_windows_trial(files,d2,ecog_grid,data_trial,0,i);
    

    %%%%%% getting online files now
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

     % trial level sort
    [data_trial] = get_spatiotemp_windows_trial(files,d2,ecog_grid,data_trial,1,i);


    %%%%%% getting batch udpated (CL2) files now
    folders = session_data(i).folders(batch_idx1);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

   % trial level sort
    [data_trial] = get_spatiotemp_windows_trial(files,d2,ecog_grid,data_trial,1,i);

end

%save alpha_dynamics_200Hz_AllDays_zscore xdata ydata labels labels_batch days -v7.3
save alpha_dynamics_1s_100Hz_AllDays_DaysLabeled_TrialLevel data_trial -v7.3


%% getting alpha osc. of hg amplitude for PAC via CNN recon



clc;clear
close all

if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    load session_data_B3_Hand
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers'))

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves/helpers'))

end

xdata={};
ydata={};
labels=[];
labels_batch=[];
days=[];

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',200);



hg_alpha_switch=false; %1 means get hG, 0 means get alpha dynamics

for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    %batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    batch_idx1 = find(folders_batch1==1);
    %     if sum(folders_batch1)==0
    %         batch_idx = find(folders_batch==1);
    %     else
    %         batch_idx = find(folders_batch1==1);
    %     end
    %     %batch_idx = [online_idx batch_idx];

    online_idx=[online_idx batch_idx];
    %batch_idx = [online_idx batch_idx_overall];


    %%%%%% get imagined data files
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx] = ...
            get_spatiotemp_windows_hg_alpha(files,d2,ecog_grid,xdata,ydata,d1);

    end

    labels = [labels; zeros(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];

    %%%%%% getting online files now
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx] = ...
            get_spatiotemp_windows_hg_alpha(files,d2,ecog_grid,xdata,ydata,d1);

    end

    labels = [labels; ones(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];


    %%%%%% getting batch udpated (CL2) files now
    folders = session_data(i).folders(batch_idx1);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx] = ...
            get_spatiotemp_windows_hg_alpha(files,d2,ecog_grid,xdata,ydata,d1);

    end

    labels = [labels; ones(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;ones(idx,1)];

end

%save alpha_dynamics_200Hz_AllDays_zscore xdata ydata labels labels_batch days -v7.3
save alpha_dynamics_hG_alpha_PAC_200Hz_AllDays_DaysLabeled xdata ydata labels labels_batch days -v7.3



%% plotting values

cl_idx= find(labels==1);
ol_idx = find(labels==0);


ol=[];
for i=1:length(ol_idx)
    tmp = cell2mat(xdata(ol_idx(i)));
    ol = [ol mean(tmp(:).^2)];
end
cl=[];
for i=1:length(cl_idx)
    tmp = cell2mat(xdata(cl_idx(i)));
    cl = [cl mean(tmp(:).^2)];
end

cl(end+1:length(ol))=NaN;
figure;boxplot([cl' ol'],'Whisker',15)
xticklabels({'CL','OL'})

[P,H,STATS]=ranksum(ol,cl)

[nanmean(cl) mean(ol)]

%%

a1k =a;
figure;
for i=1:size(a1k,1)
    tmp = squeeze(a1k(i,:,:));
    imagesc(tmp)
    pause(0.05)
end

a100=a;
figure;
for i=1:40%size(a100,1)
    tmp = squeeze(a100(i,:,:));
    imagesc(tmp)
    pause(0.1)
end

%% plotting data from Runfeng


OL=0;
CL=1;

a=[0.0,92.36183169024783,OL
    1.0,108.11812950320537,OL
    2.0,88.90075325143873,OL
    3.0,76.76341470570813,OL
    4.0,103.16923432092167,OL
    5.0,69.05802225052128,OL
    6.0,70.72858879298809,OL
    7.0,71.6972569924039,OL
    8.0,46.0521770039698,OL
    9.0,49.95963461027006,OL
    0.0,76.38132017230757,CL
    1.0,100.09545756597,CL
    2.0,83.26065216655617,CL
    3.0,74.20199629752328,CL
    4.0,98.77527423562887,CL
    5.0,55.51111725776758,CL
    6.0,57.93915679462318,CL
    7.0,64.46523458727998,CL
    8.0,42.698748308813535,CL
    9.0,46.02720458434817,CL];


ol = a(1:10,2);
cl = a(11:20,2);
days=1:10;
X = [ones(length(days),1) days'];
[B,BINT,R,RINT,STATS] = regress(ol,X);
[B1,BINT,R,RINT,STATS1] = regress(cl,X);

figure;
hold on
plot(days,ol,'.k','MarkerSize',20)
plot(days,X*B,'k','LineWidth',2)

plot(days,cl,'.b','MarkerSize',20)
plot(days,X*B1,'b','LineWidth',2)
xlabel('Days')
ylabel('1 step MSE prediction error')
set(gcf,'Color','w')
xlim([0.5 10.5])
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xticks(1:10)


figure;
boxplot([ol cl])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xticklabels({'OL','CL'})
ylabel('1 step MSE prediction error')
box off
P = signrank(ol,cl)


% cross entropy loss
OL=0;
CL=1;
a=[0.0,0.31853630978926667,OL
    1.0,1.2185988293343653,OL
    2.0,0.977496566746216,OL
    3.0,1.2986392036158376,OL
    4.0,1.0845636945841932,OL
    5.0,0.36178247683393533,OL
    6.0,0.6418155421096434,OL
    7.0,0.7917347118150979,OL
    8.0,1.1550503954475309,OL
    9.0,0.7403181394502294,OL
    0.0,0.2717143529304259,CL
    1.0,1.1196755699252263,CL
    2.0,0.8227975070622551,CL
    3.0,1.0000589976099277,CL
    4.0,0.9733150502699118,CL
    5.0,0.44613335178027963,CL
    6.0,0.7129680968167489,CL
    7.0,0.6255885338122766,CL
    8.0,0.8096929219923256,CL
    9.0,0.6309129987444196,CL]
ol = a(1:10,2);
cl=a(11:end,2);
ce_loss = mean([ol cl],2);
figure;
plot(days,ce_loss,'.','MarkerSize',20)


%% Phase amplitude coupling between hG and alpha waves (MAIN)
% do it for an example day, over all electrodes
% then branch out to all days, OL vs. CL for comparisons

% also change for PAC b/w hG and delta 


clc;clear
close all



if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    load session_data_B3_Hand
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers')

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves'))
end


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',20,'HalfPowerFrequency2',24, ...
    'SampleRate',1e3); % 8 to 10 or 0.5 to 5

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

pac_ol=[];pval_ol=[];
pac_cl=[];pval_cl=[];
pac_batch=[];pval_batch=[];
rboot_ol=[];rboot_cl=[];rboot_batch=[];
pac_raw_values={};k=1;
tic
for i=1:length(session_data)

    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    %batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    batch_idx1 = find(folders_batch1==1);
    %     if sum(folders_batch1)==0
    %         batch_idx = find(folders_batch==1);
    %     else
    %         batch_idx = find(folders_batch1==1);
    %     end
    %     %batch_idx = [online_idx batch_idx];

    online_idx=[online_idx batch_idx];
    %batch_idx = [online_idx batch_idx_overall];


    %%%%%% get imagined data files
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' OL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);



    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);

    %sum(pac_r>0.3)/253
    pval_ol(i,:) = pval;
    pac_ol(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'OL';
    pac_raw_values(k).Day = i;
    k=k+1;


    %%%%%% getting online files now
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' CL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);

    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);

    %sum(pac_r>0.3)/253
    pval_cl(i,:) = pval;
    pac_cl(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'CL';
    pac_raw_values(k).Day = i;
    k=k+1;

    %%%%%% getting batch udpated (CL2) files now
    folders = session_data(i).folders(batch_idx1);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if ~isempty(files)

        % get the phase locking value
        disp(['Processing Day ' num2str(i) ' Batch'])
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);

        % run permutation test and get pvalue for each channel
        [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);

        pval_batch(i,:) = pval;
        pac_batch(i,:) = abs(mean(pac));
        %rboot_batch(i,:,:) = rboot;
        pac_raw_values(k).pac = pac;
        pac_raw_values(k).boot = rboot;
        pac_raw_values(k).type = 'Batch';
        pac_raw_values(k).Day = i;
        k=k+1;


    else
        pac_batch(i,:)=NaN(1,253);
        pval_batch(i,:)=NaN(1,253);
    end

end

toc


cd('/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data')
save PAC_B3_Hand_rawValues_betaToHg -v7.3

%% PLOTTING CONTINUATION FROM ABOVE

% temp stuff for plotting: get good channel example of PAC
% plot significant channel on brain with preferred phase
% show how it traverses across days


% plotting example of null hypothesis testing
tmp = pac_raw_values(1).pac;
boot= pac_raw_values(1).boot;
stat=abs(mean(tmp(:,50)));
figure;
hist(boot(:,50));
vline(stat,'r')
ylabel('Count')
xlabel('PLV')
plot_beautify
xlim([0 0.7])

% no. sign channels over days after fdr correction
ol=[];cl=[];
for i=1:10
    ptmp=pval_ol(i,:);
    %[pfdr,pmask]=fdr(ptmp,0.05);
    pfdr=0.05;
    ol(i) = sum(ptmp<=pfdr)/length(ptmp);

    ptmp=pval_cl(i,:);
    %[pfdr,pmask]=fdr(ptmp,0.05);
    pfdr=0.05;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
end
% figure;
% hold on
% plot(ol,'.b','MarkerSize',20)
% plot(cl,'.r','MarkerSize',20)

days=1:10;
X = [ones(length(days),1) days'];
[B,BINT,R,RINT,STATS] = regress(ol',X);
[B1,BINT,R,RINT,STATS1] = regress(cl',X);

figure;
hold on
plot(days,ol,'.k','MarkerSize',20)
plot(days,X*B,'k','LineWidth',2)

plot(days,cl,'.b','MarkerSize',20)
plot(days,X*B1,'b','LineWidth',2)
xlabel('Days')
ylabel('Prop. of sig channels, p = 0.05 level')
set(gcf,'Color','w')
xlim([0.5 10.5])
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xticks(1:10)
legend({'','OL','','CL'})

% is diff in regression slope sig?
stat = abs(B1(2) - B(2));
boot=[];
for i=1:1000
    tmp= [ol ;cl];
    idx = rand(1,size(ol,2));
    idx(idx>0.5)=1;
    idx(idx<=0.5)=0;
    for j=1:length(idx)
        if idx(j)==1
            tmp(:,j) = flipud(tmp(:,j));
        end
    end
    ol1=tmp(1,:);
    cl1=tmp(2,:);
    [B1a] = regress(ol1',X);
    [B1b] = regress(cl1',X);
    boot(i) = abs(B1a(2)-B1b(2));
end
figure;hist(boot)
vline(stat)


% day 1, OL, channel 50 has the highest PLV: look at its relationship in hg
% vs alpha


% code for plotting phase angle and PLV on grid. Taken from ecog hand
% project code
%day_idx=1;
pac_day1 = pac_raw_values(4).pac;
plv  = abs(mean(pac_day1));
pval_day1 = pval_cl(2,:);
sig = pval_day1<=0.05;
ns = pval_day1>0.05;
pref_phase = angle(mean(pac_day1));
%subplot(1,2,1)
pax = plot_phases(pref_phase(sig));
%rose(pref_phase(sig));

% plotting plv values as an image first
pac_tmp = abs(mean(pac_day1));
ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
    pac_tmp(116:end)];
figure;
imagesc(ch_wts(ecog_grid))


% plot sig electrodes, with size denoted by PLV and color b preferred phase
% need to plot this taking into account the location of the grid and not
% just channel numbers
ch_layout=[];
for i=1:23:253
    ch_layout = [ch_layout; i:i+22 ];
end
ch_layout = (fliplr(ch_layout));

%plv(sig) = zscore(plv(sig))+4;
phMap = linspace(-pi,pi,253)';
ChColorMap = ([parula(253)]);
figure
%subplot(1,2,2)
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix, 'color', 'w','msize',2);
for j=1:length(sig)
    [xx yy]=find(ecog_grid==j);
    ch = ch_layout(xx,yy);
    if sig(j)==1
        ms = ch_wts(j)*10;
        [aa bb]=min(abs(pref_phase(j) - phMap));
        c=ChColorMap(bb,:);
        e_h = el_add(elecmatrix(ch,:), 'color', 'b','msize',ms);
    end
end
set(gcf,'Color','w')


% plotting the mean PLV over sig channels across days for OL and CL
ol_plv=[];
cl_plv=[];
ol_angle=[];
cl_angle=[];
for i=1:10
    idx = find(i==[pac_raw_values(1:end).Day]);
    for j=1:length(idx)
        if strcmp(pac_raw_values(idx(j)).type,'OL')
            tmp = pac_raw_values(idx(j)).pac;
            tmp_boot = pac_raw_values(idx(j)).boot;
            stat = abs(mean(tmp));
            pval = sum(tmp_boot>stat)./size(tmp_boot,1);
            sig = pval<=0.05;
            ns = pval>0.05;
            ol_plv(i) = mean(stat(sig));
            a = angle(mean(tmp));
            a = a(sig);
            ol_angle(i) = circ_mean(a');

        elseif strcmp(pac_raw_values(idx(j)).type,'CL')
            tmp = pac_raw_values(idx(j)).pac;
            tmp_boot = pac_raw_values(idx(j)).boot;
            stat = abs(mean(tmp));
            pval = sum(tmp_boot>stat)./size(tmp_boot,1);
            sig = pval<=0.05;
            ns = pval>0.05;
            cl_plv(i) = mean(stat(sig));
            a = angle(mean(tmp));
            a = a(sig);
            cl_angle(i) = circ_mean(a');
        end
    end
end
%
figure;plot(ol_plv)
hold on
plot(cl_plv)

ol = ol_plv;
cl = cl_plv;
% figure;
% hold on
% plot(ol,'.b','MarkerSize',20)
% plot(cl,'.r','MarkerSize',20)

days=1:10;
X = [ones(length(days),1) days'];
[B,BINT,R,RINT,STATS] = regress(ol',X);
[B1,BINT,R,RINT,STATS1] = regress(cl',X);

figure;
hold on
plot(days,ol,'.k','MarkerSize',20)
plot(days,X*B,'k','LineWidth',2)

plot(days,cl,'.b','MarkerSize',20)
plot(days,X*B1,'b','LineWidth',2)
xlabel('Days')
ylabel('Mean PLV over sig channels')
set(gcf,'Color','w')
xlim([0.5 10.5])
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xticks(1:10)
legend({'','OL','','CL'})

figure;
boxplot([ol' cl'])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xticklabels({'OL','CL'})
ylabel('Mean PLV over sig channels')
box off
P = signrank(ol,cl)


% looking at the preferred angle: always the same?


%% LOOKING AT THE AVERAGE ALPHA ERPS


%files loaded from prior sections
% do it over hand knob channel
alp_power={};
sizes=[];
for ii=1:length(files)
    disp(ii/length(files)*100)
    loaded=1;
    try
        load(files{ii})
    catch
        loaded=0;
    end

    if loaded==1

        kinax1 = find(TrialData.TaskState==1);
        kinax2 = find(TrialData.TaskState==2);
        kinax3 = find(TrialData.TaskState==3);
        kinax4 = find(TrialData.TaskState==4);

        data1 = cell2mat(TrialData.BroadbandData(kinax1)');
        l1 =  size(data1,1);
        data2 = cell2mat(TrialData.BroadbandData(kinax2)');
        l2 =  size(data2,1);
        data4 = cell2mat(TrialData.BroadbandData(kinax4)');
        l4 = size(data4,1);
        data3 = cell2mat(TrialData.BroadbandData(kinax3)');
        l3 = size(data3,1);
        sizes =[sizes; [l1 l2 l3 l4]];

        data = [data1;data2;data3;data4];
        %data = [data1;data2]; % only state 1 and 2

        % extract alpha envelope
        alp = filter(d1,data);
        alp_pow = abs(hilbert(alp));

        % % plotting example
        % figure;hold on
        % plot(alp(:,70),'LineWidth',1)
        % plot(alp_pow(:,70),'LineWidth',1)
        % h=vline(cumsum([l1 l2 l3 l4]),'m');
        % set(h,'LineWidth',1);
        % h=hline(0);
        % set(h,'Color','m')
        % set(h,'LineWidth',1);
        % ylabel('uV')
        % xlabel('Time (ms)')
        % plot_beautify
        % xticks(0:1000:15000)


        % z-score to the first 1000ms
        m = mean(alp_pow(1:1000,:),1);
        s = std(alp_pow(1:1000,:),1);
        alp_pow = (alp_pow-m)./s;

        % store
        alp_power = cat(1,alp_power,alp_pow);
    end
end


% power across trials in the various states
corr_len = median(sizes);
pow=[];ch=170;
for i=1:length(alp_power)
    tmp=alp_power{i};
    s= sizes(i,:);
    s = cumsum(s);
    s1 = tmp(1:s(1),ch);
    s1=mean(s1,1);
    s2 = tmp(s(1)+1:s(2),ch);
    s2 = mean(s2,1);
    s3 = tmp(s(2)+1:s(3),ch);
    s3 = mean(s3,1);
    s4 = tmp(s(3)+1:s(4),ch);
    s4 = mean(s4,1);
    pow = [pow; [s1 s2 s3 s4]];
end
figure;boxplot(pow)
h=hline(0,'--r');
set(h,'LineWidth',1)
xticklabels({'S1','S2','S3','S4'})
ylabel('Alpha power Z score relative to S1')
plot_beautify
title(['Channel ' num2str(ch)])

% doing it now across channels
corr_len = median(sizes);
pow_channels=[];
for ch=1:256
    if sum(ch==[118 113 108])==0
        pow=[];
        for i=1:length(alp_power)
            tmp=alp_power{i};
            s= sizes(i,:);
            s = cumsum(s);
            s1 = tmp(1:s(1),ch);
            s1=mean(s1,1);
            s2 = tmp(s(1)+1:s(2),ch);
            s2 = mean(s2,1);
            s3 = tmp(s(2)+1:s(3),ch);
            s3 = mean(s3,1);
            s4 = tmp(s(3)+1:s(4),ch);
            s4 = mean(s4,1);
            pow = [pow; [s1 s2 s3 s4]];
        end
        pow_channels =[pow_channels;median(pow)];
    end
end
figure;boxplot(pow_channels)
h=hline(0,'--r');
set(h,'LineWidth',1)
xticklabels({'S1','S2','S3','S4'})
ylabel('Alpha power Z score relative to S1')
plot_beautify
title(['All Channels']);




%% SIGNIFICANT HG ERPS




%files loaded from prior sections
% do it over hand knob channel
alp_power={};
sizes=[];
for ii=1:length(files)
    disp(ii/length(files)*100)
    loaded=1;
    try
        load(files{ii})
    catch
        loaded=0;
    end

    if loaded==1

        kinax1 = find(TrialData.TaskState==1);
        kinax2 = find(TrialData.TaskState==2);
        kinax3 = find(TrialData.TaskState==3);
        kinax4 = find(TrialData.TaskState==4);

        data1 = cell2mat(TrialData.BroadbandData(kinax1)');
        l1 =  size(data1,1);
        data2 = cell2mat(TrialData.BroadbandData(kinax2)');
        l2 =  size(data2,1);
        data4 = cell2mat(TrialData.BroadbandData(kinax4)');
        l4 = size(data4,1);
        data3 = cell2mat(TrialData.BroadbandData(kinax3)');
        l3 = size(data3,1);
        sizes =[sizes; [l1 l2 l3 l4]];

        data = [data1;data2;data3;data4];
        %data = [data1;data2]; % only state 1 and 2

        % extract alpha envelope
        alp = filter(d1,data);
        alp_pow = abs(hilbert(alp));

        % % plotting example
        % figure;hold on
        % plot(alp(:,70),'LineWidth',1)
        % plot(alp_pow(:,70),'LineWidth',1)
        % h=vline(cumsum([l1 l2 l3 l4]),'m');
        % set(h,'LineWidth',1);
        % h=hline(0);
        % set(h,'Color','m')
        % set(h,'LineWidth',1);
        % ylabel('uV')
        % xlabel('Time (ms)')
        % plot_beautify
        % xticks(0:1000:15000)


        % z-score to the first 1000ms
        m = mean(alp_pow(1:1000,:),1);
        s = std(alp_pow(1:1000,:),1);
        alp_pow = (alp_pow-m)./s;

        % store
        alp_power = cat(1,alp_power,alp_pow);
    end
end


%% getting the alpha signals for the 7 DoF dataset

clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data

xdata={};
ydata={};
labels=[];
labels_batch=[];
days=[];

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',200);



hg_alpha_switch=false; %1 means get hG, 0 means get alpha dynamics
session_data = session_data([1:9 11]); % removing bad days

for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    if i==3 || i==6 || i==8
        folders_pm = strcmp(session_data(i).AM_PM,'pm');
        folders_batch(folders_pm==0)=0;
        if i==8
            idx = find(folders_batch==1);
            folders_batch(idx(3:end))=0;
        end
    else
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_batch(folders_am==0) = 0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    online_idx=[online_idx batch_idx];
    %batch_idx = [online_idx batch_idx_overall];


    %%%%%% get imagined data files
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',...
            folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end
    load(files{1})
    ecog_grid=TrialData.Params.ChMap;

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx] = get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata);

    end

    labels = [labels; zeros(idx,1)];
    days = [days;ones(idx,1)*i];
    %labels_batch = [labels_batch;zeros(idx,1)];

    %%%%%% getting online files now
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date, 'Robot3DArrow',...
            folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx] = get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata);

    end

    labels = [labels; ones(idx,1)];
    days = [days;ones(idx,1)*i];
    %labels_batch = [labels_batch;zeros(idx,1)];


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
    %  if hg_alpha_switch
    %     [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);
    %
    % else
    %     [xdata,ydata,idx] = get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata);
    %
    % end
    %
    % labels = [labels; ones(idx,1)];
    % days = [days;ones(idx,1)*i];
    % labels_batch = [labels_batch;ones(idx,1)];

end

%save alpha_dynamics_200Hz_AllDays_zscore xdata ydata labels labels_batch days -v7.3
%save alpha_dynamics_hG_200Hz_AllDays_DaysLabeled_B1_7DoF xdata ydata labels labels_batch days -v7.3
save alpha_dynamics_hG_200Hz_All5Days_DaysLabeled_B1_7DoF xdata ydata labels labels_batch days -v7.3



%% PAC hG and alpha waves in 7DoF Dataset (B3)
% do it for an example day, over all electrodes
% then branch out to all days, OL vs. CL for comparisons



clc;clear
close all



if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    %load session_data_B3_Hand
    load session_data_B3
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers')

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves'))
end


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

pac_ol=[];pval_ol=[];
pac_cl=[];pval_cl=[];
pac_batch=[];pval_batch=[];
rboot_ol=[];rboot_cl=[];rboot_batch=[];
pac_raw_values={};k=1;
tic
for i=13:15%length(session_data)

    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    %batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    batch_idx1 = find(folders_batch1==1);
    %     if sum(folders_batch1)==0
    %         batch_idx = find(folders_batch==1);
    %     else
    %         batch_idx = find(folders_batch1==1);
    %     end
    %     %batch_idx = [online_idx batch_idx];

    %online_idx=[online_idx batch_idx];
    online_idx=[online_idx ];
    %batch_idx = [online_idx batch_idx_overall];


    %%%%%% get imagined data files
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' OL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);

    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    %[pfdr,pthresh]=fdr(pval,0.05);
    %sum(pthresh)

    %sum(pac_r>0.3)/253
    pval_ol(i,:) = pval;
    pac_ol(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'OL';
    pac_raw_values(k).Day = i;
    k=k+1;


    %%%%%% getting online files now
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' CL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);


    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    %[pfdr,pthresh]=fdr(pval,0.05);
    %sum(pthresh)

    %sum(pac_r>0.3)/253
    pval_cl(i,:) = pval;
    pac_cl(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'CL';
    pac_raw_values(k).Day = i;
    k=k+1;

    %%%%%% getting batch udpated (CL2) files now
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if ~isempty(files)

        % get the phase locking value
        disp(['Processing Day ' num2str(i) ' Batch'])
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);

        % run permutation test and get pvalue for each channel
        [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
        %[pfdr,pthresh]=fdr(pval,0.05);
        %sum(pthresh)

        pval_batch(i,:) = pval;
        pac_batch(i,:) = abs(mean(pac));
        %rboot_batch(i,:,:) = rboot;
        pac_raw_values(k).pac = pac;
        pac_raw_values(k).boot = rboot;
        pac_raw_values(k).type = 'Batch';
        pac_raw_values(k).Day = i;
        k=k+1;


    else
        pac_batch(i,:)=NaN(1,253);
        pval_batch(i,:)=NaN(1,253);
    end

end

toc

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
save PAC_B3_7DoF_rawValues -v7.3

%cd('/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data')
%save PAC_B3_Hand_rawValues -v7.3


% temp plotting the results
aa= sum(pval_ol'<0.01)/253;
bb= sum(pval_cl'<0.01)/253;
cc= sum(pval_batch'<0.01)/253;
figure;plot(aa);
hold on
plot(bb)
plot(cc)


%% PAC hG and alpha waves in 7DoF Dataset (B1)
% do it for an example day, over all electrodes
% then branch out to all days, OL vs. CL for comparisons



clc;clear
close all



if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    %load session_data_B3_Hand
    load session_data
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers')

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves'))
end


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

session_data = session_data([1:9 11]);
pac_ol=[];pval_ol=[];
pac_cl=[];pval_cl=[];
pac_batch=[];pval_batch=[];
rboot_ol=[];rboot_cl=[];rboot_batch=[];
pac_raw_values={};k=1;

for i=1:length(session_data)


    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    if i==3 || i==6 || i==8
        folders_pm = strcmp(session_data(i).AM_PM,'pm');
        folders_batch(folders_pm==0)=0;
        if i==8
            idx = find(folders_batch==1);
            folders_batch(idx(3:end))=0;
        end
    else
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_batch(folders_am==0) = 0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


    %     if sum(folders_batch1)==0
    %         batch_idx = find(folders_batch==1);
    %     else
    %         batch_idx = find(folders_batch1==1);
    %     end
    %     %batch_idx = [online_idx batch_idx];

    %online_idx=[online_idx batch_idx];
    online_idx=[online_idx ];
    %batch_idx = [online_idx batch_idx_overall];


    %%%%%% get imagined data files
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' OL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2,1);

    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase,1);
    %[pfdr,pthresh]=fdr(pval,0.05);
    %sum(pthresh)

    %sum(pac_r>0.3)/253
    pval_ol(i,:) = pval;
    pac_ol(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'OL';
    pac_raw_values(k).Day = i;
    k=k+1;


    %%%%%% getting online files now
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' CL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2,1);


    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase,1);
    %[pfdr,pthresh]=fdr(pval,0.05);
    %sum(pthresh)

    %sum(pac_r>0.3)/253
    pval_cl(i,:) = pval;
    pac_cl(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'CL';
    pac_raw_values(k).Day = i;
    k=k+1;

    %%%%%% getting batch udpated (CL2) files now
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    if ~isempty(files)

        % get the phase locking value
        disp(['Processing Day ' num2str(i) ' Batch'])
        [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2,1);

        % run permutation test and get pvalue for each channel
        [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase,1);
        %[pfdr,pthresh]=fdr(pval,0.05);
        %sum(pthresh)

        pval_batch(i,:) = pval;
        pac_batch(i,:) = abs(mean(pac));
        %rboot_batch(i,:,:) = rboot;
        pac_raw_values(k).pac = pac;
        pac_raw_values(k).boot = rboot;
        pac_raw_values(k).type = 'Batch';
        pac_raw_values(k).Day = i;
        k=k+1;


    else
        pac_batch(i,:)=NaN(1,253);
        pval_batch(i,:)=NaN(1,253);
    end

end



cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save PAC_B1_7DoF_rawValues_128_grid -v7.3

%cd('/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data')
%save PAC_B3_Hand_rawValues -v7.3


% temp plotting the results
aa= sum(pval_ol'<0.05)/128;
bb= sum(pval_cl'<0.05)/128;
cc= sum(pval_batch'<0.05)/128;
figure;plot(aa);
hold on
plot(bb)
plot(cc)

