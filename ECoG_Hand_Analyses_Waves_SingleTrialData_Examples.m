%% DETECTION OF TRAVELING WAVES IN SINGLE TRIAL DATA


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
    %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))

end

% load cl2 trials from last day
%i=length(session_data);
i=1;
folders_imag =  strcmp(session_data(i).folder_type,'I');
folders_online = strcmp(session_data(i).folder_type,'O');
folders_batch = strcmp(session_data(i).folder_type,'B');
folders_batch1 = strcmp(session_data(i).folder_type,'B1');
imag_idx = find(folders_imag==1);
online_idx = find(folders_online==1);
batch_idx = find(folders_batch==1);
batch_idx1 = find(folders_batch1==1);
online_idx=[online_idx batch_idx];


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);
d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',200);
hilbert_flag=1;


%%%%%% get imagined data files
folders = session_data(i).folders(imag_idx);
day_date = session_data(i).Day;
files=[];
for ii=1:length(folders)
    folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
    %cd(folderpath)
    files = [files;findfiles('mat',folderpath)'];
end

stats_ol = rotational_waves_stats(files,d2,hilbert_flag,ecog_grid);



%%%%%% get batch data files %%%%%
folders = session_data(1).folders(online_idx);
day_date = session_data(1).Day;
files=[];
for ii=1:length(folders)
    folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %cd(folderpath)
    files = [files;findfiles('mat',folderpath)'];
end

stats_cl = rotational_waves_stats(files,d2,hilbert_flag,ecog_grid);


x=[];
for i = 1:length(stats_ol)
    %tmp = stats_ol(i).size;
    tmp = stats_ol(i).corr;
    %x(i,:) = smooth(tmp(1:2.2e3),50);
    %tmp = smooth(tmp,50);
    %x= [x; nanmedian(tmp)];
    x=[x;sum(isnan(tmp))/length(tmp)];
end
%figure;plot(mean(x,1))
xol=x;

x=[];
for i = 1:length(stats_cl)
    tmp = stats_cl(i).corr;
    %tmp = smooth(tmp,10);
    %x= [x; nanmedian(tmp)];
    %x(i,:) = smooth(tmp(1:500),50);
    x=[x;sum(isnan(tmp))/length(tmp)];
end
xcl=x;


[p,h] = ranksum(xol,xcl)
figure;boxplot([xol xcl],'notch','off')


%% SAME BUT IN B1

% arrow tasks


clc;clear
close all

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

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',9, ...
    'SampleRate',1e3);
d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',9, ...
    'SampleRate',200);
hilbert_flag=1;


folders={'20240515', '20240517', '20240614', ...
    '20240619', '20240621', '20240626',...
    '20240710','20240712','20240731'};
xol_days=[];
xcl_days=[];
for days=1:length(folders)-1
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
        if strcmp(subfoldername(3).name,'Imagined')
            imag_idx=[imag_idx j];
        elseif strcmp(subfoldername(3).name,'BCI_Fixed')
            online_idx=[online_idx j];
        end
    end
    imag_idx_main=imag_idx(1:3);
    online_idx_main=online_idx(1:3);

    for folder=1:3

        imag_idx = imag_idx_main(folder);
        online_idx = online_idx_main(folder);



        %%%%%% get imagined data files
        files=[];
        for ii=1:length(imag_idx)
            imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');
            files = [files;findfiles('mat',imag_folderpath)'];
        end
        %files=files(1:21);
        stats_ol = rotational_waves_stats(files,d2,hilbert_flag,ecog_grid);



        %%%%%% get batch data files %%%%%
        files=[];
        for ii=1:length(online_idx)
            imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
            files = [files;findfiles('mat',imag_folderpath)'];
        end
        %files=files(1:21);
        stats_cl = rotational_waves_stats(files,d2,hilbert_flag,ecog_grid);


        x=[];
        for i = 1:length(stats_ol)
            tmp = stats_ol(i).corr;
            %tmp = stats_ol(i).corr;
            %x(i,:) = smooth(tmp(1:2.2e3),50);
            %tmp = smooth(tmp,50);
            %x= [x; nanmedian(tmp)];
            %x=[x;sum(isnan(tmp))/length(tmp)];

            %tmp = stats_ol(i).size;
            x=[x;nanmean(tmp)];
        end
        %figure;plot(mean(x,1))
        xol=x;


        x=[];
        for i = 1:length(stats_cl)
            tmp = stats_cl(i).corr;
            %tmp = smooth(tmp,10);
            %x= [x; nanmedian(tmp)];
            %x(i,:) = smooth(tmp(1:500),50);
            %x=[x;sum(isnan(tmp))/length(tmp)];

            %tmp = stats_cl(i).size;
            % if length(tmp)>1000
            %     x=[x;mean(tmp)];
            % end
            x=[x;nanmean(tmp)];
        end
        xcl=x;

        %xol_days(days) = median(xol);
        %xcl_days(days) = median(xcl);

        xol_days = [xol_days median(xol)];
        xcl_days = [xcl_days median(xcl)];
    end
end

[p,h] = ranksum(xol_days,xcl_days)
figure;boxplot([xol_days' xcl_days'],'notch','off')

%% USING A LINEAR MODEL TO PREDICT TEMPORALLY 1 STEP INTO FUTURE

clc;clear;
%cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/')
cd('/media/user/Data/ECoG_BCI_TravelingWave_Data/')

% load data
%load('alpha_dynamics_B1_253_Arrow_200Hz_AllDays_DaysLabeled_ArtifactCorr')
load('alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec')

parpool('threads')

recon_ol=[];
recon_cl=[];
for i=1:10
    disp(num2str(i))

    % take day 1 data alone
    idx = find(days==i);
    labels_day = labels(idx);
    x_day = xdata(idx);
    y_day = ydata(idx);

    ol = find(labels_day==0);
    cl = find(labels_day==1);

    % ol recon error
    x = x_day(ol);
    y = y_day(ol);
    recon_ol(i) = lin_pred_model(x,y);

    % cl recon error
    x = x_day(cl);
    y = y_day(cl);
    recon_cl(i) = lin_pred_model(x,y);

end

figure;
plot(recon_ol);
hold on
plot(recon_cl);


%% B3: CHECK PLANAR WAVES IN 6 BY 6 MINIGRID ROLLING AROUND THE OVERALL GRID
% (MAIN, ARROW OR HAND)

clc;clear;
close all
if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    load session_data_B3_Hand
    %load session_data_B3
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\'))

else
    %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
    cd(root_path)
    load session_data_B3_Hand
    %load session_data_B3
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
close all

% load cl2 trials from last day
%i=length(session_data);
xol_days=[];
xcl_days=[];
stats_ol_days={};
stats_cl_days={};
len_days = min(11,length(session_data));
stats_ol_hg_days={};
stats_cl_hg_days={};
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
        grid_layout,elecmatrix,bpFilt);
    stats_ol_days{days}=stats_ol;
    stats_ol_hg_days{days} = stats_ol_hg;



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

    len = min(150,length(files));
    idx=randperm(length(files),len);
     [stats_cl,stats_cl_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
        grid_layout,elecmatrix,bpFilt);
    stats_cl_days{days}=stats_cl;
    stats_cl_hg_days{days}=stats_cl_hg;

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

save B3_waves_hand_stability_Muller_hG -v7.3

figure;
plot(xol_days)
hold on
plot(xcl_days)

figure;
boxplot([xol_days' xcl_days'])
[p,h] = signrank(xol_days,xcl_days)
[h,p,tb,st]=ttest(xol_days,xcl_days)

% plotting back
xol=[];xcl=[];
for i=1:length(stats_ol_days)
    tmp0=stats_ol_days{i};
    x=[];
    for j=1:length(tmp0)
        tmp_og = (tmp0(j).stab);
        tmp=zscore(tmp_og);
        [out,st,stp] = wave_stability_detect(tmp);
        % s=[];
        % for k=1:length(st)
        %     s(k) = median(tmp_og(st(k):stp(k)));
        % end
        % x(j) = median(s);


        %x(j) = median(out);
        %x(j) = sum(out)/length(tmp);

        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(j) = f*d;%duty_cycle

        
    end
    xol(i) = nanmean(x);

    tmp0=stats_cl_days{i};
    x=[];
    for j=1:length(tmp0)
        tmp_og = (tmp0(j).stab);
        tmp=zscore(tmp_og);
        [out,st,stp] = wave_stability_detect(tmp);
        % s=[];
        % for k=1:length(st)
        %     s(k) = median(tmp_og(st(k):stp(k)));
        % end
        % x(j) = median(s);
        %x(j) = median(out);
        %x(j) = sum(out)/length(tmp);

        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(j) = f*d;%duty_cycle
        
        %x(j) = median(tmp_og);
    end
    xcl(i) = nanmean(x);
end

figure;
plot(xol);hold on
plot(xcl)


figure;
%boxplot([xol' xcl']*20)
boxplot([xol' xcl'])
plot_beautify
xticks(1:2)
xticklabels({'OL','CL'})
ylabel('Duty Cycle')
[p,h] = signrank(xol,xcl)
[h,p,tb,st]=ttest(xol,xcl)

% when plotting duty cycle analyses,
% B3_waves_hand_stability_Muller is significant
% B3_waves_hand_stability approaches signficiance (median) and is
% significat in t-test



% looking at hg differences between two conditions, wave vs. non wave
res=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    D_wave=[];D_nonwave=[];
    for tid0=1:12
        act1=[];act1_nonwave=[];
         for i=1:length(stats_cl_hg)
                if stats_cl_hg(i).target_id ==tid0
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    act1 = [act1;tmp];

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act1_nonwave = [act1_nonwave;tmp];
                end
         end
        for tid=tid0+1:12            
            act2=[];act2_nonwave=[];
            for i=1:length(stats_cl_hg)
                % if stats_cl_hg(i).target_id ==tid0
                %     tmp = stats_cl_hg(i).hg_wave;
                %     tmp = cell2mat(tmp');
                %     act1 = [act1;tmp];
                % 
                %     tmp = stats_cl_hg(i).hg_nonwave;
                %     tmp = cell2mat(tmp');
                %     act1_nonwave = [act1_nonwave;tmp];
                % end
                if stats_cl_hg(i).target_id ==tid
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    act2 = [act2;tmp];

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act2_nonwave = [act2_nonwave;tmp];
                end
            end
            d1 = mahal2(act1,act2,2);
            d2 = mahal2(act1_nonwave,act2_nonwave,2);
            D_wave = [D_wave d1];
            D_nonwave = [D_nonwave d2];
        end
    end
    res(days,:) = [median(D_wave) median(D_nonwave)];
end
figure;
boxplot(res)
[p,h] = signrank(res(:,1),res(:,2))
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Pairwise Mahalanobis dist.')
title('hG Decoding Information')
plot_beautify

% examining hg amplitude during waves vs. without
res=[];
hg_wave=[];hg_nonwave=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};

    for i=1:length(stats_cl_hg)
        tmp = cell2mat(stats_cl_hg(i).hg_wave');
        tmp = mean(tmp(:));
        hg_wave = [hg_wave tmp];

        tmp = cell2mat(stats_cl_hg(i).hg_nonwave');
        tmp = mean(tmp(:));
        hg_nonwave = [hg_nonwave tmp];
    end
end
figure;
res=[hg_wave' hg_nonwave'];
boxplot(res)

% potential tile name
% High performance closed-loop brain computer interface control is
% associated with mu traveling waves

% build a classifer to differentiate conditions by their traveling waves
% right thumb vs. power grasp 
% 
% A=[];
% B=[];
% for days =1:length(stats_cl_hg_days)
%     stats_cl_hg = stats_cl_hg_days{days};
%     for i=1:length(stats_cl_hg)
%         if stats_cl_hg(i).target_id==1
%             tmp = stats_cl_hg(i).mu_wave;
%             for j=1:length(tmp)
%                 a = tmp{j};
%                 a = mean(a,1);
%                 A = [A ;a];
%             end
%         elseif stats_cl_hg(i).target_id==2
%             tmp = stats_cl_hg(i).mu_wave;
%             for j=1:length(tmp)
%                 b = tmp{j};
%                 b = mean(b,1);
%                 B = [B ;b];
%             end
%         end
%     end
% end
% 
% if size(B,1) > size(A,1)
%     len = size(A,1);
%     idx = randperm(size(B,1),len);
%     B = B(idx,:);
% elseif size(B,1) < size(A,1)
%     len = size(B,1);
%     idx = randperm(size(A,1),len);
%     A = A(idx,:);
% end
% 
% X = [real(A) imag(A) ; real(B) imag(B)];
% Y = [zeros(size(A,1),1); ones(size(B,1),1)];
% 
% cv = cvpartition(Y, 'HoldOut', 0.2); % Hold out 30% for testing
% 
% idxTrain = cv.training;
% idxTest = cv.test;
% 
% XTrain = X(idxTrain,:);
% YTrain = Y(idxTrain,:);
% XTest = X(idxTest,:);
% YTest = Y(idxTest,:);
% 
% % 2. Train the Linear SVM model
% % Use the 'Linear' kernel option to specify a linear SVM.
% disp('Training linear SVM model...');
% Mdl = fitcsvm(XTrain, YTrain, 'KernelFunction', 'Linear', 'Standardize', true);
% disp('Model trained.');
% 
% % 3. Use the trained model to predict new data
% % The 'predict' function returns the predicted labels.
% [label, score] = predict(Mdl, XTest);
% 
% % 4. Evaluate the model (optional)
% % Calculate the accuracy of the predictions on the test set.
% confusion_matrix = confusionmat(YTest, label);
% accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));
% fprintf('Accuracy on test set: %.2f%%\n', accuracy * 100);
% 
% 
% mdl = fitclinear(X,Y,'CrossVal','on','Leaveout','on');


% examinining whether there are divergence and curl differences between OL
% and CL across days and trials from the stable wave epochs




%% B1 CHECK PLANAR WAVES IN 6 BY 6 MINIGRID ROLLING AROUND THE OVERALL GRID
% (MAIN)


clc;clear;
close all
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
    'SampleRate',1e3);
d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',10, ...
    'SampleRate',50);
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);
hilbert_flag=1;


imaging_B3_waves;close all

folders={'20240515', '20240517', '20240614', ...
    '20240619', '20240621', '20240626',...
    '20240710','20240712','20240731'};

%folders = folders(1:8);

xol_days=[];
xcl_days=[];
stats_ol_days={};
stats_cl_days={};
stats_ol_hg_days={};
stats_cl_hg_days={};
for days=1:8%length(folders)

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
        if strcmp(subfoldername(3).name,'Imagined')
            imag_idx=[imag_idx j];
        elseif strcmp(subfoldername(3).name,'BCI_Fixed')
            online_idx=[online_idx j];
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

    len = min(175,length(files));
    idx=randperm(length(files),len);
    [stats_ol,stats_ol_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
        grid_layout,elecmatrix,bpFilt);
    stats_ol_days{days}=stats_ol;
    stats_ol_hg_days{days} = stats_ol_hg;


    %%%%%% get online data files %%%%%
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    len = min(175,length(files));
    idx=randperm(length(files),len);
    [stats_cl,stats_cl_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
        grid_layout,elecmatrix,bpFilt);
    stats_cl_days{days}=stats_cl;
    stats_cl_hg_days{days}=stats_cl_hg;

    
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

    %%%% performing PAC during wave vs. non wave epochs
    % whether per channel the phase angle is preserved across epochs during
    % wave vs. non wave epochs
    % pac_wave=[];
    % pac_nonwave=[];
    % for i=1:length(stats_ol_hg)
    %     hg = stats_ol_hg(i).hg_wave;        
    %     for j=1:length(hg)
    %         h = hg{j}';            
    %         % d = exp(1i * (angle(h) - angle(m)) );
    %         % d= angle(mean(d,1));
    %         % pac_wave = cat(1,pac_wave,d);
    %         [P, f ,Phi, lambda, Xhat, z0, Z,rf]=dmd_alg(h,50,0,20);
    %     end
    % end
    
    % whether the same phase angle across all electrodes


    




end

%save B1_waves_stability_New -v7.3 % 50Hz, removing last 400ms in fitering step
save B1_waves_stability_hG -v7.3 % 50Hz, removing last 400ms in fitering step


figure;
plot(xol_days)
hold on
plot(xcl_days)

figure;
boxplot([xol_days' xcl_days']*1)
[p,h] = signrank(1*xol_days,1*xcl_days)
[h,p,tb,st]=ttest(xol_days,xcl_days)


% plotting back
xol=[];xcl=[];
for i=1:length(stats_ol_days)
    tmp0=stats_ol_days{i};
    x=[];
    for j=1:length(tmp0)
        tmp_og = (tmp0(j).stab);
        tmp=zscore(tmp_og);
        [out,st,stp] = wave_stability_detect(tmp);
        % s=[];
        % for k=1:length(st)
        %     s(k) = median(tmp_og(st(k):stp(k)));
        % end
        % x(j) = median(s);


        %x(j) = median(out);
        %x(j) = sum(out)/length(tmp);

        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(j) = f*d;%duty_cycle

        
    end
    xol(i) = nanmean(x);

    tmp0=stats_cl_days{i};
    x=[];
    for j=1:length(tmp0)
        tmp_og = (tmp0(j).stab);
        tmp=zscore(tmp_og);
        [out,st,stp] = wave_stability_detect(tmp);
        % s=[];
        % for k=1:length(st)
        %     s(k) = median(tmp_og(st(k):stp(k)));
        % end
        % x(j) = median(s);
        %x(j) = median(out);
        %x(j) = sum(out)/length(tmp);

        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(j) = f*d;%duty_cycle
        
        %x(j) = median(tmp_og);
    end
    xcl(i) = nanmean(x);
end

figure;
plot(xol);hold on
plot(xcl)


figure;
boxplot([xol' xcl']*1e3)
boxplot([xol' xcl'])
[p,h] = signrank(xol,xcl)
[h,p,tb,st]=ttest(xol,xcl)


% when plotting duty cycle,
% B1_waves_stability_Muller is significant
% have to do analyses for B1_waves_stability (run it overnight)


% looking at hg differences between two conditions, wave vs. non wave
res=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    D_wave=[];D_nonwave=[];
    for tid0=1:7
        for tid=tid0+1:7
            act1=[];act2=[];
            act1_nonwave=[];act2_nonwave=[];
            for i=1:length(stats_cl_hg)
                if stats_cl_hg(i).target_id ==tid0
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    act1 = [act1;tmp];

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act1_nonwave = [act1_nonwave;tmp];
                end
                if stats_cl_hg(i).target_id ==tid
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    act2 = [act2;tmp];

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act2_nonwave = [act2_nonwave;tmp];
                end
            end
            d1 = mahal2(act1,act2,2);
            d2 = mahal2(act1_nonwave,act2_nonwave,2);
            D_wave = [D_wave d1];
            D_nonwave = [D_nonwave d2];
        end
    end
    res(days,:) = [median(D_wave) median(D_nonwave)];
end

figure;
boxplot(res)
[p,h] = signrank(res(:,1),res(:,2))
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Pairwise Mahalanobis dist.')
title('hG Decoding Information')
plot_beautify


% DIFFERENCES BETWEEN OL AND CL STABLE EPOCHS IN TERMS OF ROTATIONS AND
% CONTRACTIONS/EXPANSIONS
xol=[];xcl=[];
for i=1:length(stats_ol_days)
    tmp0=stats_ol_days{i};
    x=[];
    for j=1:length(tmp0)
        tmp_og = (tmp0(j).stab);
        tmp=zscore(tmp_og);
        [out,st,stp] = wave_stability_detect(tmp);
        for k=1:length(st)
            tmp = tmp0(j).vec_field(st(k):stp(k),:,:);
            tmp = squeeze(mean(tmp,1));
            tmp = tmp./max(abs(tmp(:)));
            [XX,YY] = meshgrid( 1:size(tmp,2), 1:size(tmp,1) );
            M=real(tmp);
            N=imag(tmp);
            [C] = divergence(XX,YY,M,N);
            x = [x max(abs(C(:)))];
        end
    end
    xol(i) = nanmean(x);

    tmp0=stats_cl_days{i};
    x=[];
    for j=1:length(tmp0)
        tmp_og = (tmp0(j).stab);
        tmp=zscore(tmp_og);
        [out,st,stp] = wave_stability_detect(tmp);
        for k=1:length(st)
            tmp = tmp0(j).vec_field(st(k):stp(k),:,:);
            tmp = squeeze(mean(tmp,1));
            tmp = tmp./max(abs(tmp(:)));
            [XX,YY] = meshgrid( 1:size(tmp,2), 1:size(tmp,1) );
            M=real(tmp);
            N=imag(tmp);
            [C] = divergence(XX,YY,M,N);
            x = [x max(abs(C(:)))];
        end
    end    
    xcl(i) = nanmean(x);
end

figure;
plot(xol);hold on
plot(xcl)


figure;
boxplot([xol' xcl']*1e3)
boxplot([xol' xcl'])
[p,h] = signrank(xol,xcl)
[h,p,tb,st]=ttest(xol,xcl)


for i=1:size(tmp,1)
    tmp0 = squeeze(tmp(i,:,:));
    figure;
    M = real(tmp0);
    N = imag(tmp0);
    quiver(XX,YY,M,N);axis tight
end

figure;
[XX,YY] = meshgrid( 1:size(tmp,2), 1:size(tmp,1) );
tmp = tmp./max(abs(tmp(:)));
M = real(tmp);
N = imag(tmp);
quiver(XX,YY,M,N);axis tight
[C,cav] = curl(XX,YY,M,N);
figure;imagesc(C)

%% B6 CHECK PLANAR WAVES IN 6 BY 6 MINIGRID ROLLING AROUND THE OVERALL GRID
%(MAIN)

clc;clear
close all

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
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/helpers'))

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



%
% % hand
% folders={'20250624', '20250703', ...
%      '20250827', '20250903', '20250917','20250924'}; %20250708 has only imagined

% robot3DArrow
folders = {'20250530','20250610','20250624','20250703','20250708','20250717',...
    '20250917','20250924'};

imaging_B3_waves;close all

hilbert_flag=1;
xol_days=[];
xcl_days=[];
stats_ol_days={};
stats_cl_days={};
stats_ol_hg_days={};
stats_cl_hg_days={};
for days=1:length(folders)-1

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
    [stats_ol,stats_ol_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
        grid_layout,elecmatrix,bpFilt);
    stats_ol_days{days}=stats_ol;
    stats_ol_hg_days{days} = stats_ol_hg;


    %%%%%% get online data files %%%%%
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    len = min(200,length(files));
    idx=randperm(length(files),len);
    [stats_cl,stats_cl_hg] = planar_waves_stats(files(idx),d2,hilbert_flag,ecog_grid,...
        grid_layout,elecmatrix,bpFilt);
    stats_cl_days{days}=stats_cl;
    stats_cl_hg_days{days}=stats_cl_hg;

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

    %save B6_waves_stability -v7.3 % 50Hz, removing last 400ms in fitering step

end

figure;
plot(xol_days)
hold on
plot(xcl_days)

figure;
boxplot([xol_days' xcl_days']*1)
[p,h] = signrank(1*xol_days,1*xcl_days)
[h,p,tb,st]=ttest(xol_days,xcl_days)

save B6_waves_stability_Muller_hG -v7.3 % 50Hz, removing last 400ms in fitering step

% plotting back
xol=[];xcl=[];
for i=1:length(stats_ol_days)
    tmp0=stats_ol_days{i};
    x=[];
    for j=1:length(tmp0)
        tmp_og = (tmp0(j).stab);
        tmp=zscore(tmp_og);
        [out,st,stp] = wave_stability_detect(tmp);
        % s=[];
        % for k=1:length(st)
        %     s(k) = median(tmp_og(st(k):stp(k)));
        % end
        % x(j) = median(s);


        %x(j) = median(out);
        %x(j) = sum(out)/length(tmp);

        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(j) = f*d;%duty_cycle

        
    end
    xol(i) = nanmean(x);

    tmp0=stats_cl_days{i};
    x=[];
    for j=1:length(tmp0)
        tmp_og = (tmp0(j).stab);
        tmp=zscore(tmp_og);
        [out,st,stp] = wave_stability_detect(tmp);
        % s=[];
        % for k=1:length(st)
        %     s(k) = median(tmp_og(st(k):stp(k)));
        % end
        % x(j) = median(s);
        %x(j) = median(out);
        %x(j) = sum(out)/length(tmp);

        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = median(out) * 20/1e3; %duration in s
        x(j) = f*d;%duty_cycle
        
        %x(j) = median(tmp_og);
    end
    xcl(i) = nanmean(x);
end

figure;
plot(xol);hold on
plot(xcl)


figure;
%boxplot([xol' xcl']*20)
boxplot([xol' xcl'])
[p,h] = signrank(xol,xcl)
[h,p,tb,st]=ttest(xol,xcl)

%when comparing the duty cycle, 
% B6_waves_stability_Muller trending significant (also needs one more
% session to be analyzed, load the data from box)
% B6_waves_stability is significant




% looking at hg differences between two conditions, wave vs. non wave
res=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    D_wave=[];D_nonwave=[];
    for tid0=1:7
        for tid=tid0+1:7
            act1=[];act2=[];
            act1_nonwave=[];act2_nonwave=[];
            for i=1:length(stats_cl_hg)
                if stats_cl_hg(i).target_id ==tid0
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    act1 = [act1;tmp];

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act1_nonwave = [act1_nonwave;tmp];
                end
                if stats_cl_hg(i).target_id ==tid
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    act2 = [act2;tmp];

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act2_nonwave = [act2_nonwave;tmp];
                end
            end
            d1 = mahal2(act1,act2,2);
            d2 = mahal2(act1_nonwave,act2_nonwave,2);
            D_wave = [D_wave d1];
            D_nonwave = [D_nonwave d2];
        end
    end
    res(days,:) = [median(D_wave) median(D_nonwave)];
end
figure;boxplot(res)
[p,h] = signrank(res(:,1),res(:,2))



%% RUN COMPLEX VALUED ICA ON THE STABLE EPOCH VECTOR FIELDS

clc;
clear;
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
load load B1_waves_stability_hG

% get the mean vector field whenever a stable epoch is found 
% so do cICA on each day. Find components within top 20 with curl greater
% than 0.5. Compare its weight for each of the OL and CL cases
res_days=[];
for days =5:length(stats_ol_days)
    disp(num2str(days))
    stats_ol = stats_ol_days{days};
    stats_cl = stats_cl_days{days};
    x=[];
    for i=1:length(stats_ol)
        tmp = (stats_ol(i).stab);
        v = stats_ol(i).vec_field;
        tmp=zscore(tmp(1:end));
        [out,st,stp] = wave_stability_detect(tmp);
        for j=1:length(out)
            a = squeeze(v(st(j):stp(j),:,:));
            a = squeeze(mean(a,1));
            x= [x a(:)];
        end
    end

    l = size(x,2);
    for i=1:length(stats_cl)
        tmp = (stats_cl(i).stab);
        v = stats_cl(i).vec_field;
        tmp=zscore(tmp(1:end));
        [out,st,stp] = wave_stability_detect(tmp);
        for j=1:length(out)
            a = squeeze(v(st(j):stp(j),:,:));
            a = squeeze(mean(a,1));
            x= [x a(:)];
        end
    end

    [W, min_cost, Ahat, Shat] = complex_ICA_EBM(x);
    z = W*x;

    res=[];
    for i=1:15 % first 15 components
        xph = Ahat(:,i);
        xph = reshape(xph,11,23);
        xph = xph./max(abs(xph(:)));
        M = real(xph);
        N = imag(xph);
        [XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );
        [d,c]= curl(XX,YY,M,N);
        [div]=divergence(XX,YY,M,N);
        if max(d(:)) > 0.5
            a = abs(z(i,:));
            res=  [res;[ mean(a(1:l)) mean(a(l+1:end))]];
        end
    end
    res_days = [res_days;mean(res,1)];
end



figure;
boxplot(res_days)



% compute variances per components
% model is X = A*s where s is sources (along each row)

% preprocess
X=x;
X = X-mean(X,2);
% [N,T] = size(X);
% % remove DC
% Xmean=mean(X,2);
% X = X - Xmean*ones(1,T);    
% % spatio pre-whitening 1 
% R = X*X'/T;                 
% P = inv_sqrtmH(R);  %P = inv(sqrtm(R));
% X = P*X;

vaf=[];
norm_x = norm(X,'fro');
for i=1:size(Ahat,1)
    tmp = Ahat(:,i) * Shat(i,:);
    vaf(i) = norm(tmp,'fro') / norm_x;
end



