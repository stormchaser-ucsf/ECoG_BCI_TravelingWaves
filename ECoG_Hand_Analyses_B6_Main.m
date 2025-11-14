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


%% INIT
clc;clear

%addpath('/home/user/Documents/MATLAB')
addpath('/home/user/Documents/MATLAB/CircStat2012a')
addpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/helpers')
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/wave-matlab-master/wave-matlab-master'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves'))
cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')

%% JUST EXTRACTING OL CROSS VAL ACC AND CL BEST PERFORMANCE BLOCKS
% across days

% 
% folders={'20250624', '20250703', ...
%     '20250827', '20250903', '20250917','20250924'}; %20250708 has only imagined
% folders = folders(4:end);

%root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B6';
cd(root_path)
imagined_folders={};k=1;
online_folders={};kk=1;
D = dir(root_path);
for i=3:length(D)-2
    filepath = fullfile(root_path,D(i).name);
    D1 = dir(filepath);    
    for j=3:length(D1)
        if ~isempty(regexp(D1(j).name,'HandImagined'))
            imagined_folders(i).Day = D(i).name;k=k+1;
        elseif ~isempty(regexp(D1(j).name,'HandOnline'))
            online_folders(i).Day = D(i).name;kk=kk+1;
        end
    end
end

ol_acc_days=[];
cl_acc_days=[];
load('ECOG_Grid_8596_000067_B3.mat')
plot_true=true;
% doing OL
for i=1:length(imagined_folders)
    folder = imagined_folders(i).Day;
    files=[];
    if ~isempty(folder)
        disp(['Processing ' folder])
        filepath = fullfile(root_path,imagined_folders(i).Day,'HandImagined');
        files=findfiles('.mat',filepath,1)';
        files = remove_kf_params(files);
        condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);
        [acc_imagined,train_permutations,~,bino_pdf] = ...
            accuracy_imagined_data_Hand_B3(condn_data, 10);
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
        ol_acc_days=cat(3,ol_acc_days,acc_imagined);
    end
end

tmp=[];
for i=1:size(ol_acc_days,3)
    a= squeeze(ol_acc_days(:,:,i));
    a = mean(diag(a));
    tmp =[tmp a];
end
figure;
plot(tmp)
ol_curve=tmp;

% doing CL now
for i=1:length(online_folders)
    folder = online_folders(i).Day;
    files=[];
    if ~isempty(folder)
        disp(['Processing ' folder])
        filepath = fullfile(root_path,online_folders(i).Day,'HandOnline');
        files=findfiles('.mat',filepath,1)';
        files = remove_kf_params(files);
        %condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);
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
            box on
            title(['CL Accuracy of ' num2str(100*mean(diag(acc_online)))])
            xticks(1:12)
            yticks(1:12)
            xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
                'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
            yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
                'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        end
        cl_acc_days=cat(3,cl_acc_days,acc_online);
    end
end

tmp=[];
for i=1:size(cl_acc_days,3)
    a= squeeze(cl_acc_days(:,:,i));
    a = mean(diag(a));
    tmp =[tmp a];
end
figure;
plot(tmp)
cl_curve=tmp;


% doing CL now, best blocks
cl_acc_days=[];best_block=[];
for i=1:length(online_folders)
    folder = online_folders(i).Day;
    files=[];
    if ~isempty(folder)
        disp(['Processing ' folder])
        filepath = fullfile(root_path,online_folders(i).Day,'HandOnline');
        acc=[];
        D2 = dir(filepath);
        acc_online_blocks=[];
        for j=3:length(D2)
            subfilepath = fullfile(filepath,D2(j).name,'BCI_Fixed');
            files=findfiles('.mat',subfilepath,1)';
            [acc_online,acc_online_bin,trial_len,bino_pdf] = ...
                accuracy_online_data_Hand(files,12);        
            acc=[acc,mean(diag(acc_online))];
            acc_online_blocks = cat(3,acc_online_blocks,acc_online);
        end
        [aa bb]=max(acc);
        cl_acc_days=cat(3,cl_acc_days,aa);
        acc_online=squeeze(acc_online_blocks(:,:,bb));
        best_block=[best_block bb/length(acc)];
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
            box on
            title(['CL Accuracy of ' num2str(100*mean(diag(acc_online)))])
            xticks(1:12)
            yticks(1:12)
            xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
                'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
            yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
                'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        end
    end
end

tmp=[];
for i=1:size(cl_acc_days,3)
    a= squeeze(cl_acc_days(:,:,i));
    a = mean(diag(a));
    tmp =[tmp a];
end
figure;
plot(tmp)
cl_curve=tmp;

% plotting
days=1:length(cl_curve);
X = [ones(length(days),1) days'];
[B,BINT,R,RINT,STATS] = regress(ol_curve',X);
[B1,BINT,R,RINT,STATS1] = regress(cl_curve',X);
ol = ol_curve;
cl=cl_curve;
figure;
hold on
plot(days,ol,'.k','MarkerSize',20)
plot(days,X*B,'k','LineWidth',2)

plot(days,cl,'.b','MarkerSize',20)
plot(days,X*B1,'b','LineWidth',2)
xlabel('Days')
ylabel('Decoding Accuracy')
set(gcf,'Color','w')
xlim([0.5 3.5])
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xticks(0.5:3.5)
legend({'','OL','','CL'})


%% SESSION DATA FOR HAND EXPERIMENTS 

clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd(root_path)

%day1
session_data(1).Day = '20230421';
session_data(1).folders = {'135818','140708','141436','142253','142936',...
    '144212','144820','145512','150244','151515'};
session_data(1).folder_type={'I','I','I','I','I','O','O','O','O',...
    'B'};
session_data(1).AM_PM = {'pm','pm','pm','pm','pm','pm','pm','pm','pm','pm'};

%day2
session_data(2).Day = '20230428';
session_data(2).folders = {'134224','135107','140247','140924','141455',...
    '142804','143248','143755','144547',...
    '145812','150746'};
session_data(2).folder_type={'I','I','I','I','I','O','O','O','O',...
    'B','B'};
session_data(2).AM_PM = {'pm','pm','pm','pm','pm','pm','pm','pm','pm','pm','pm'};

%day3 -> this has to be reuploaded from box
session_data(3).Day = '20230505';
session_data(3).folders = {'110553','111203','111808','112152','113118','113451',...
    '114430','114929','115412','115953',...
    '120747','121326'};
session_data(3).folder_type={'I','I','I','I','I','I','O','O','O','O'...
    'B','B'};
session_data(3).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am'};

%day4    -> this has to be reuploaded from box
session_data(4).Day = '20230510';
session_data(4).folders = {'103338','104124','114511','105158','105560','110015',...    
    '110943','111353','111812','112222',...
    '113101','113848'};
session_data(4).folder_type={'I','I','I','I','I','I','O','O','O','O'...
    'B','B'};
session_data(4).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am'};

%day5 
session_data(5).Day = '20230519';
session_data(5).folders = {'104503','105053','105633','110010','110535','110913',...    
    '112006','112551','113226','113919',...
    '114925','115323'};
session_data(5).folder_type={'I','I','I','I','I','I','O','O','O','O'...
    'B','B'};
session_data(5).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am'};

%day6 -> this has to be reuploaded from box
session_data(6).Day = '20240522';
session_data(6).folders = {'103410','104032','104409','105436','105817','110148','110624',...    
    '112404','112809','113728','114116','114913',...
    '120606','120855','121135',...
    '121959','122204','122358'};
session_data(6).folder_type={'I','I','I','I','I','I','I',...
    'O','O','O','O','O'...
    'B','B','B',...
    'B1','B1','B1'};
session_data(6).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am','am','am','am','am','am','am'};

%day7 -> needs to be re uploaded to box as its not there, entries below
%have to edited
session_data(7).Day = '20240524';
session_data(7).folders = {'103410','104032','104409','105436','105817','110148','110624',...    
    '112404','112809','113728','114116','114913',...
    '120606','120855','121135',...
    '121959','122204','122358'};
session_data(7).folder_type={'I','I','I','I','I','I','I',...
    'O','O','O','O','O'...
    'B','B','B',...
    'B1','B1','B1'};
session_data(7).AM_PM = {'am','am','am','am','am','am','am','am','am','am',...
    'am','am','am','am','am','am','am','am'};



save session_data_B1_Hand session_data


%% OPEN LOOP OSCILLATION CLUSTERS
% get all the files from a particular day

%filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20250315\HandImagined';

%20240515, 20240517, 20240614, 20240619, 20240621, 20240626

%filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20230519\HandOnline';
%filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20240517\Robot3DArrow';
%filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20250120\RealRobotBatch';
%filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/20250708/HandImagined';
filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/20250708/HandImagined';

%filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20250120\RealRobotBatch';

load('ECOG_Grid_8596_000067_B3.mat')

files = findfiles('.mat',filepath,1)';
files = remove_kf_params(files);

%elec_list=1:256;
elec_list = [137	143	148	152	155	23
159	160	30	28	25	21
134	140	170	174	178	182
49	45	41	38	35	163
221	62	59	56	52	48
205	208	211	214	218	222];

bad_ch=[108 113 118];
%bad_ch=[];
osc_clus=[];
stats=[];
pow_freq=[];
ffreq=[];
for ii=1:length(files)
    disp(ii/length(files)*100)
    loaded=true;
    try
        load(files{ii})
    catch
        loaded=false;
    end

    % if sum(TrialData.TargetID == [1 3 7 ]) > 0
    %     hand=true;
    % else 
    %     hand=false;
    % end
    hand=true;

    if loaded && hand
        data_trial = (TrialData.BroadbandData');

        % run power spectrum and 1/f stats on a single trial basis
        task_state = TrialData.TaskState;
        kinax = find(task_state==3);
        data = cell2mat(data_trial(kinax));

        %data=data(:,elec_list);
        spectral_peaks=[];
        stats_tmp=[];
        parfor i=1:size(data,2)
            %disp(i)
            x = data(:,i);
            [Pxx,F] = pwelch(x,1024,512,1024,1e3);
            pow_freq = [pow_freq;Pxx' ];
            ffreq = [ffreq ;F'];
            idx = logical((F>0) .* (F<=40));
            %idx = logical((F>0) .* (F<=150));
            %idx = logical((F>65) .* (F<=150));
            F1=F(idx);
            F1=log2(F1);
            power_spect = Pxx(idx);
            power_spect = log2(power_spect);
            %[bhat p wh se ci t_stat]=robust_fit(F1,power_spect,1);
            %tb=fitlm(F1,power_spect,'RobustOpts','huber');
            %stats_tmp = [stats_tmp tb.Coefficients.pValue(2)];
            %bhat = tb.Coefficients.Estimate;

            [b, stats] = robustfit(F1,power_spect, 'huber',...
                [], 'on');
            bhat = b;

            
            x = [ones(length(F1),1) F1];
            yhat = x*bhat;

            % %plot
            % figure;
            % plot(F1,power_spect,'LineWidth',1);
            % hold on
            % plot(F1,yhat,'LineWidth',1);

            % get peaks in power spectrum at specific frequencies
            power_spect = zscore(power_spect - yhat);
            [aa bb]=findpeaks(power_spect);
            peak_loc = bb(find(power_spect(bb)>1));
            freqs = 2.^F1(peak_loc);
            pow = power_spect(peak_loc);

            %store
            spectral_peaks(i).freqs = freqs;
            spectral_peaks(i).pow = pow;
        end


        % getting oscillation clusters
        osc_clus_tmp=[];
        for f=2:40 % 40 earlier
        %for f=66:150 % 40 earlier
            ff = [f-1 f+1];
            tmp=0;ch_tmp=[];
            for j=1:length(spectral_peaks)
                if sum(j==bad_ch)==0
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
end


% plot oscillation clusters
f=2:40;
%f=2:150;
%f=66:150;
figure;
hold on
plot(f,osc_clus,'Color',[.5 .5 .5 .5],'LineWidth',.5)
plot(f,median(osc_clus,1),'b','LineWidth',2)


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
        if sum( (f>=7) .* (f<=10) ) >= 1
            ch_idx=[ch_idx i];
        end
    end
end
length(ch_idx)/(length(spectral_peaks)-3);
I = zeros(256,1);
I(ch_idx)=1;
figure;imagesc(I(ecog_grid))

% get all electrodes within 16 and 20Hz
ch_idx=[];
for i=1:length(spectral_peaks)
    if sum(i==bad_ch)==0
        f = spectral_peaks(i).freqs;
        if sum( (f>=15) .* (f<=18) ) >= 1
            ch_idx=[ch_idx i];
        end
    end
end
length(ch_idx)/(length(spectral_peaks)-3);
I = zeros(256,1);
I(ch_idx)=1;
figure;imagesc(I(ecog_grid))

%% loading data, filtering and extracting epochs for use in a COMPLEX CNN AE
% MAIN


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

xdata={};
ydata={};
labels=[];
labels_batch=[];
days=[];
mvmt_labels=[];
trial_number=[];
data={};

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7.5,'HalfPowerFrequency2',9.5, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7.5,'HalfPowerFrequency2',9.5, ...
    'SampleRate',200);


% hand
folders={'20250624', '20250703', ...
     '20250827', '20250903', '20250917','20250924'}; %20250708 has only imagined

% robot3DArrow
folders = {'20250530','20250610','20250624','20250703','20250708','20250717',...
    '20250917','20250924'};


hg_alpha_switch=false; %1 means get hG, 0 means get alpha dynamics

for i=1:length(folders)

    folderpath = fullfile(root_path,folders{i});
    D= dir(folderpath);
    D = D(3:end);
    imag_idx=[];
    online_idx=[];
    for j=1:length(D)        
        if strcmp(D(j).name,'HandImagined')
            imag_idx=[imag_idx j];
        elseif strcmp(D(j).name,'HandOnline')
            online_idx=[online_idx j];
        end
    end

    % folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
    % D= dir(folderpath);
    % D = D(3:end);
    % imag_idx=[];
    % online_idx=[];
    % for j=1:length(D)
    %     subfoldername = dir(fullfile(folderpath,D(j).name));
    %     if length(subfoldername)>2
    %         if strcmp(subfoldername(3).name,'Imagined')
    %             imag_idx=[imag_idx j];
    %         elseif strcmp(subfoldername(3).name,'BCI_Fixed')
    %             online_idx=[online_idx j];
    %         end
    %     end
    % end

    %%%%%% get imagined data files    
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name);        
        files = [files;findfiles('.mat',imag_folderpath,1)'];
    end
    files = remove_kf_params(files);

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx,trial_idx] = ...
            get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata,1);

    end

    labels = [labels; zeros(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];
    mvmt_labels= [mvmt_labels;trial_idx];

    %%%%%% getting online files now
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name);
        files = [files;findfiles('.mat',imag_folderpath,1)'];
    end
    files = remove_kf_params(files);

    if hg_alpha_switch
        [xdata,ydata,idx,trial_idx] = ...
            get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata,1);

    else
        [xdata,ydata,idx,trial_idx] = ...
            get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata,1);
    end

    labels = [labels; ones(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];
    mvmt_labels= [mvmt_labels;trial_idx];
end


for i=1:length(xdata)
    disp(i/length(xdata)*100)
    tmp=xdata{i};
    tmp = single(tmp);
    xdata{i}=tmp;

    tmp=ydata{i};
    tmp = single(tmp);
    ydata{i}=tmp;
end

%save alpha_dynamics_200Hz_AllDays_zscore xdata ydata labels labels_batch days -v7.3
save alpha_dynamics_B6_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_Hand xdata ydata labels labels_batch days -v7.3



%% Phase amplitude coupling between hG and alpha waves (MAIN)
% do it for an example day, over all electrodes
% then branch out to all days, OL vs. CL for comparisons

% also change for PAC b/w hG and delta 


clc;clear
close all



if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B6';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')    
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers')

else
    %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    root_path='/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
    cd(root_path)
    %load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/helpers'))

end


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7.5,'HalfPowerFrequency2',9.5, ...
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

% % hand folders
folders={'20250624', '20250703', ...
    '20250827', '20250903', '20250917','20250924'}; %20250708 has only imagined
folders = folders(4:end);
parpool('threads')

%robot3d arrow folders
%folders = {'20250530','20250610','20250624','20250703','20250708','20250717','20250917'};

for i=1:length(folders)

    disp(['Processing day ' folders{i}])

    folderpath = fullfile(root_path,folders{i});
    D= dir(folderpath);
    D = D(3:end);
    imag_idx=[];
    online_idx=[];
    for j=1:length(D)
        if strcmp(D(j).name,'HandImagined')
            imag_idx=[imag_idx j];
        elseif strcmp(D(j).name,'HandOnline')
            online_idx=[online_idx j];
        end
    end

    % folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
    % D= dir(folderpath);
    % D = D(3:end);
    % imag_idx=[];
    % online_idx=[];
    % for j=1:length(D)
    %     subfoldername = dir(fullfile(folderpath,D(j).name));
    %     if length(subfoldername)>2
    %         if strcmp(subfoldername(3).name,'Imagined')
    %             imag_idx=[imag_idx j];
    %         elseif strcmp(subfoldername(3).name,'BCI_Fixed')
    %             online_idx=[online_idx j];
    %         end
    %     end
    % end


    %%%%%% get imagined data files    
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name);        
        files = [files;findfiles('.mat',imag_folderpath,1)'];
    end
    files = remove_kf_params(files);


    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' OL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);



    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase);
    %sum(pval<=0.05)

    %sum(pac_r>0.3)/253
    pval_ol(i,:) = pval;
    pac_ol(i,:) = abs(mean(pac));
    pac_raw_values(k).pac = pac;
    pac_raw_values(k).boot = rboot;
    pac_raw_values(k).type = 'OL';
    pac_raw_values(k).Day = i;
    k=k+1;


    %%%%%% getting online files now     
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name);
        files = [files;findfiles('.mat',imag_folderpath,1)'];
    end
    files = remove_kf_params(files);
    

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
end

toc


%cd('/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data')
%save PAC_B6_Hand_muToHg_7pt5To9pt5Hz_500Iter_Hand_woutDays1n3 -v7.3

%% PLOTTING, CONTINUATION FROM ABOVE


imaging_B1_253;
close all

%cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%load('PAC_B1_253Grid_7DoF_rawValues_alphaToHg.mat')


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

% num. sign channels over days after fdr correction
ol=[];cl=[];
for i=1:size(pval_cl,1)
    ptmp=pval_ol(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);
    %pfdr=0.05;
    ol(i) = sum(ptmp<=pfdr)/length(ptmp);

    ptmp=pval_cl(i,:);
    [pfdr,pmask]=fdr(ptmp,0.05);
    %pfdr=0.05;
    cl(i) = sum(ptmp<=pfdr)/length(ptmp);
end
% figure;
% hold on
% plot(ol,'.b','MarkerSize',20)
% plot(cl,'.r','MarkerSize',20)

days=1:size(pval_cl,1);
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
xlim([0.5 size(pval_cl,1)+0.5])
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xticks(1:size(pval_cl,1))
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

figure;boxplot([ol' cl'])


% day 1, OL, channel 50 has the highest PLV: look at its relationship in hg
% vs alpha


% code for plotting phase angle and PLV on grid. Taken from ecog hand
% project code
%day_idx=1;
pac_day1 = pac_raw_values(12).pac;
plv  = abs(mean(pac_day1));
pval_day1 = pval_cl(6,:);
[pfdr,pval1]=fdr(pval_day1,0.05);pfdr
%pfdr=0.05;
sig = pval_day1<=pfdr;
ns = pval_day1>pfdr;
pref_phase = angle(mean(pac_day1));
%subplot(1,2,1)
pax = plot_phases(pref_phase(sig));
%rose(pref_phase(sig));

% plotting plv values as an image first
pac_tmp = abs(mean(pac_day1));
ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
    pac_tmp(116:end)];
sig1 = [sig(1:107) 0 sig(108:111) 0  sig(112:115) 0 ...
    sig(116:end)];
figure;
imagesc(ch_wts(ecog_grid))
figure;
imagesc(ch_wts(ecog_grid).*sig1(ecog_grid))

% plot sig electrodes, with size denoted by PLV and color b preferred phase
% need to plot this taking into account the location of the grid and not
% just channel numbers

%plv(sig) = zscore(plv(sig))+4;
phMap = linspace(-pi,pi,253)';
ChColorMap = ([parula(253)]);
figure
%subplot(1,2,2)
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix, 'color', 'w','msize',2);
% elecmatrix1 = [elecmatrix(1:107,:); zeros(1,3); elecmatrix(108:111,:); zeros(1,3) ; ...
%     elecmatrix(112:115,:) ;zeros(1,3); elecmatrix(116:end,:)];
ch_wts1=pac_tmp;
for j=1:253
    if sig(j)==1
        ms = ch_wts1(j)*20;
        %[aa bb]=min(abs(pref_phase(j) - phMap));
        c=ChColorMap(bb,:);
        e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',ms);
    end
end
set(gcf,'Color','w')


% plotting the mean PLV over sig channels across days for OL and CL
ol_plv=[];
cl_plv=[];
ol_angle=[];
cl_angle=[];
for i=1:length(unique([pac_raw_values(1:end).Day]))
    idx = find(i==[pac_raw_values(1:end).Day]);
    for j=1:length(idx)
        if strcmp(pac_raw_values(idx(j)).type,'OL')
            tmp = pac_raw_values(idx(j)).pac;
            tmp_boot = pac_raw_values(idx(j)).boot;
            stat = abs(mean(tmp));
            pval = sum(tmp_boot>stat)./size(tmp_boot,1);
            sig = pval<=0.05;
            ns = pval>0.05;
            %ol_plv(i) = mean(stat(sig));
            ol_plv(i) = mean(stat());
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
            %cl_plv(i) = mean(stat(sig));
            cl_plv(i) = mean(stat());
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

days=1:length(unique([pac_raw_values(1:end).Day]));
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
xlim([0.5 6.5])
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xticks(1:6)
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



%% RUNNING LDA TO GET DECODING PERFORMANCE

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
addpath('C:\Users\nikic\Documents\MATLAB')
foldernames = {'20250315'}'%{'20220302','20220223'};20220302 is online hand...amaze
cd(root_path)

%filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\\HandImagined';

imagined_files=[];
for i=1:length(foldernames)
    if i==1
       folderpath = fullfile(root_path, foldernames{i},'HandOnline');
    else
       folderpath = fullfile(root_path, foldernames{i},'Hand');
    end
    %folderpath = fullfile(root_path, foldernames{i},'HandImagined');

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

res_overall=[];
for iter=1:5
    disp(iter)


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
    idx = randperm(length(imagined_files),round(0.8*length(imagined_files)));
    train_files = imagined_files(idx);
    I = ones(length(imagined_files),1);
    I(idx)=0;
    test_files = imagined_files(find(I==1));

    for i=1:length(train_files)
        %disp(i/length(train_files)*100)
        try
            load(train_files{i})
            file_loaded = true;
        catch
            file_loaded=false;
            disp(['Could not load ' files{j}]);
        end


        if file_loaded
            action = TrialData.TargetID;
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax = find(kinax==3);
            temp = cell2mat(features(kinax));
            temp = temp(:,1:end);

            % get the smoothed and pooled data
            % get smoothed delta hg and beta features
            % new_temp=[];
            % [xx yy] = size(TrialData.Params.ChMap);
            % for k=1:size(temp,2)
            %     tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            %     tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            %     tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            %     tmp4 = temp(641:768,k);tmp4 = tmp4(TrialData.Params.ChMap);
            %     tmp5 = temp(385:512,k);tmp5 = tmp5(TrialData.Params.ChMap);
            %     pooled_data=[];
            %     for i=1:2:xx
            %         for j=1:2:yy
            %             delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
            %             beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
            %             hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
            %             lg = (tmp4(i:i+1,j:j+1));lg=mean(lg(:));
            %             %alp = (tmp5(i:i+1,j:j+1));alp=mean(alp(:));
            %             pooled_data = [pooled_data; delta;beta;hg];
            %             %pooled_data = [pooled_data; hg];
            %         end
            %     end
            %     new_temp= [new_temp pooled_data];
            % end
            % temp=new_temp;
            % data_seg = temp(1:end,:); % only high gamma
            %data_seg = mean(data_seg,2);

            % get the features in B3 format            
            temp = temp([257:512 1025:1280 1537:1792],:);
            bad_ch = [108 113 118 ];
            good_ch = ones(size(temp,1),1);
            for ii=1:length(bad_ch)                
                bad_ch_tmp = bad_ch(ii)+(256*[0 1 2]);
                good_ch(bad_ch_tmp)=0;
            end
            temp = temp(logical(good_ch),:);

            % 2-norm
            for ii=1:size(temp,2)
                temp(:,ii) = temp(:,ii)./norm(temp(:,ii));
            end

            data_seg = temp(1:end,:);
            

            if action ==1
                D1i = cat(2,D1i,data_seg);
                %D1f = cat(2,D1f,feat_stats1);
            elseif action ==2
                D2i = cat(2,D2i,data_seg);
                %D2f = cat(2,D2f,feat_stats1);
            elseif action ==3
                D3i = cat(2,D3i,data_seg);
                %D3f = cat(2,D3f,feat_stats1);
            elseif action ==4
                D4i = cat(2,D4i,data_seg);
                %D4f = cat(2,D4f,feat_stats1);
            elseif action ==5
                D5i = cat(2,D5i,data_seg);
                %D5f = cat(2,D5f,feat_stats1);
            elseif action ==6
                D6i = cat(2,D6i,data_seg);
                %D6f = cat(2,D6f,feat_stats1);
            elseif action ==7
                D7i = cat(2,D7i,data_seg);
                %D7f = cat(2,D7f,feat_stats1);
            elseif action ==8
                D8i = cat(2,D8i,data_seg);
                %D7f = cat(2,D7f,feat_stats1);
            elseif action ==9
                D9i = cat(2,D9i,data_seg);
            elseif action ==10
                D10i = cat(2,D10i,data_seg);
            end
        end
    end

    data=[];
    Y=[];
    data=[data cell2mat(D1i)]; Y=[Y;0*ones(size(cell2mat(D1i),2),1)];
    data=[data cell2mat(D2i)];  Y=[Y;1*ones(size(cell2mat(D2i),2),1)];
    data=[data cell2mat(D3i)];  Y=[Y;2*ones(size(cell2mat(D3i),2),1)];
    data=[data cell2mat(D4i)];  Y=[Y;3*ones(size(cell2mat(D4i),2),1)];
    data=[data cell2mat(D5i)];  Y=[Y;4*ones(size(cell2mat(D5i),2),1)];
    %data=[data cell2mat(D6i)];  Y=[Y;5*ones(size(cell2mat(D6i),2),1)];
    %data=[data cell2mat(D7i)];  Y=[Y;6*ones(size(cell2mat(D7i),2),1)];
    %data=[data cell2mat(D8i)];  Y=[Y;7*ones(size(cell2mat(D8i),2),1)];
    %data=[data cell2mat(D9i)];  Y=[Y;6*ones(size(cell2mat(D9i),2),1)];
    %data=[data cell2mat(D10i)];  Y=[Y;7*ones(size(cell2mat(D10i),2),1)];
    data=data';

    % run LDA
    W = LDA(data,Y);

    % run it on the held out files and get classification accuracies
    acc=zeros(size(W,1));
    for i=1:length(test_files)
        %disp(i/length(test_files)*100)
        try
            load(test_files{i})
            file_loaded = true;
        catch
            file_loaded=false;
            disp(['Could not load ' files{j}]);
        end


        if file_loaded
            action = TrialData.TargetID;
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax = find(kinax==3);
            temp = cell2mat(features(kinax));
            temp = temp(:,1:end);

            % % get the smoothed and pooled data
            % % get smoothed delta hg and beta features
            % new_temp=[];
            % [xx yy] = size(TrialData.Params.ChMap);
            % for k=1:size(temp,2)
            %     tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            %     tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            %     tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            %     tmp4 = temp(641:768,k);tmp4 = tmp4(TrialData.Params.ChMap);
            %     tmp5 = temp(385:512,k);tmp5 = tmp5(TrialData.Params.ChMap);
            %     pooled_data=[];
            %     for i=1:2:xx
            %         for j=1:2:yy
            %             delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
            %             beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
            %             hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
            %             lg = (tmp4(i:i+1,j:j+1));lg=mean(lg(:));
            %             %alp = (tmp5(i:i+1,j:j+1));alp=mean(alp(:));
            %             pooled_data = [pooled_data; delta;beta;hg];
            %             %pooled_data = [pooled_data; hg];
            %         end
            %     end
            %     new_temp= [new_temp pooled_data];
            % end
            % temp=new_temp;
            % data_seg = temp(1:end,:); % only high gamma
            %data_seg = mean(data_seg,2);

            % get in b3 format
            temp = temp([257:512 1025:1280 1537:1792],:);
            bad_ch = [108 113 118 ];
            good_ch = ones(size(temp,1),1);
            for ii=1:length(bad_ch)
                bad_ch_tmp = bad_ch(ii)+(256*[0 1 2]);
                good_ch(bad_ch_tmp)=0;
            end
            temp = temp(logical(good_ch),:);

            % 2-norm
            for ii=1:size(temp,2)
                temp(:,ii) = temp(:,ii)./norm(temp(:,ii));
            end

            data_seg = temp(1:end,:);
        end
        data_seg = data_seg';

        % run it thru the LDA
        L = [ones(size(data_seg,1),1) data_seg] * W';

        % get classification prob
        P = exp(L) ./ repmat(sum(exp(L),2),[1 size(L,2)]);

        %average prob
        decision = nanmean(P(1:end,:));
        %decision = P;
        [aa bb]=max(decision);

        % correction for online trials
%         if TrialData.TargetID==9
%             TrialData.TargetID = 7;
%         elseif TrialData.TargetID==10
%             TrialData.TargetID = 8;
%         end


        % store results
        if TrialData.TargetID <=5
            acc(TrialData.TargetID,bb) = acc(TrialData.TargetID,bb)+1;
        end
    end

    for i=1:length(acc)
        acc(i,:)= acc(i,:)/sum(acc(i,:));
    end
    %figure;imagesc(acc)
    %diag(acc)
    %mean(ans)

    res_overall(iter,:,:)=acc;

end

acc=squeeze(nanmean(res_overall,1));
figure;imagesc(acc)
diag(acc)
mean(ans)
colormap bone
caxis([0 1])
set(gcf,'Color','w')
title(['Av. Classif. Acc of ' num2str(mean(diag(acc))) '%'])
xticks(1:5)
yticks(1:5)
xticklabels({'Thumb','Index','Middle','Ring','Little'})
yticklabels({'Thumb','Index','Middle','Ring','Little'})
set(gca,'FontSize',14)

%% PERFORMANCE IMAGINED - ONLINE- BATCH 

clc;clear
close all
clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=10;
plot_true=false;
acc_batch_days_overall=[];
folders={'20240515', '20240517', '20240614', '20240619', '20240621', '20240626'};

for i=1:length(folders)

    %folderpath = fullfile(root_path,'B1_253',folders{i},'Robot3DArrow');
    folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
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

    %%%%%% get imagined data files
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);
    
    condn_data1={};k=1;
    for j=1:length(condn_data)
        if condn_data(j).targetID <=7
            condn_data1(k).neural = condn_data(j).neural;
            condn_data1(k).trial_type = condn_data(j).trial_type;
            condn_data1(k).targetID = condn_data(j).targetID;
            k=k+1;
        end
    end
    condn_data = condn_data1;
    clear condn_data1
    

    % get cross-val classification accuracy    
    [acc_imagined,train_permutations,~,bino_pdf,bino_pdf_chance] =...
        accuracy_imagined_data(condn_data, iterations);    
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_imagined)))])
        xticks(1:12)
        yticks(1:12)
        xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
    end
    acc_imagined_days(:,i) = diag(acc_imagined);

    
     %%%%%% getting online files now
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);
    
    condn_data1={};k=1;
    for j=1:length(condn_data)
        if condn_data(j).targetID <=7
            condn_data1(k).neural = condn_data(j).neural;
            condn_data1(k).trial_type = condn_data(j).trial_type;
            condn_data1(k).targetID = condn_data(j).targetID;
            k=k+1;
        end
    end
    condn_data = condn_data1;
    clear condn_data1

    % get the classification accuracy 
    acc_online = accuracy_online_data(files(end-42:end));

    % [acc_online,train_permutations,~,bino_pdf,bino_pdf_chance] =...
    %     accuracy_imagined_data(condn_data, iterations);    
    % acc_online=squeeze(nanmean(acc_online,1));

    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
        title(['Accuracy of ' num2str(100*mean(diag(acc_online)))])
        xticks(1:12)
        yticks(1:12)
        xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
        yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
            'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
    end
    acc_online_days(:,i) = diag(acc_online);

    % 
    % %%%%%% classification accuracy for batch data
    % folders = session_data(i).folders(batch_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %     %cd(folderpath)
    %     files = [files;findfiles('',folderpath)'];
    % end
    % 
    % % get the classification accuracy
    % acc_batch = accuracy_online_data_Hand(files,12);
    % if plot_true
    %     figure;imagesc(acc_batch)
    %     colormap bone
    %     clim([0 1])
    %     set(gcf,'color','w')
    %     title(['Accuracy of ' num2str(100*mean(diag(acc_batch)))])
    %     xticks(1:12)
    %     yticks(1:12)
    %     xticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
    %         'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})
    %     yticklabels({'Thumb','Index','Middle','Ring','Pinky','Power',...
    %         'Pinch','Tripod','Wrist In','Wrist Out','Wrist Flex','Wrist Extend'})       
    % 
    % end
    % acc_batch_days(:,i) = diag(acc_batch);
    % acc_batch_days_overall(:,:,i)=acc_batch;
end

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


%% Phase amplitude coupling between hG and alpha waves (MAIN)
% do it for an example day, over all electrodes
% then branch out to all days, OL vs. CL for comparisons

% also change for PAC b/w hG and delta 


clc;clear
close all



if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    load session_data_B1_Hand
    addpath 'C:\Users\nikic\Documents\MATLAB'
    %load('ECOG_Grid_8596_000067_B3.mat')
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
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
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
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2,1);



    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase,1);

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
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2,1);

    % run permutation test and get pvalue for each channel
    [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase,1);

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
save PAC_B3_Hand_rawValues_betaToHg_15To20Hz -v7.3


%% Phase amplitude coupling between hG and alpha waves (MAIN) 253 channel
% 253 channel grid 
% do it for an example day, over all electrodes
% then branch out to all days, OL vs. CL for comparisons
    
% also change for PAC b/w hG and delta 


clc;clear
close all



if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    load session_data_B1_Hand
    addpath 'C:\Users\nikic\Documents\MATLAB'
    %load('ECOG_Grid_8596_000067_B3.mat')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers')

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    %load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves'))
end


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',9, ...
    'SampleRate',1e3); % 7 to 9 or 0.5 to 4

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);


folders={'20240515', '20240517', '20240614', '20240619', '20240621', '20240626'};

pac_ol=[];pval_ol=[];
pac_cl=[];pval_cl=[];
pac_batch=[];pval_batch=[];
rboot_ol=[];rboot_cl=[];rboot_batch=[];
pac_raw_values={};k=1;
tic
for i=1:length(folders)

    folderpath = fullfile(root_path,'B1_253',folders{i},'Robot3DArrow');
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



    %%%%%% get imagined data files
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');
        files = [files;findfiles('mat',imag_folderpath)'];
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
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
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



if ~ispc
    cd('/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data/B1_253')
else
    cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
end
save PAC_B1_253Grid_7DoF_rawValues_alphaToHg -v7.3




toc

%% PLOTTING CONTINUATION FROM ABOVE

% temp stuff for plotting: get good channel example of PAC
% plot significant channel on brain with preferred phase
% show how it traverses across days






%% GETTING ALPHA WAVE FOR ARROW DATA, 253 grid



clc;clear
close all

if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')    
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\ECOG_Grid_8596_000067_B3.mat')
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
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',9, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',9, ...
    'SampleRate',200);


folders={'20240515', '20240517', '20240614', '20240619', '20240621', '20240626'};



hg_alpha_switch=false; %1 means get hG, 0 means get alpha dynamics

for i=1:length(folders)

    folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
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

    %%%%%% get imagined data files    
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');        
        files = [files;findfiles('mat',imag_folderpath)'];
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
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx] = get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata);

    end

    labels = [labels; ones(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];


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
    % if hg_alpha_switch
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
save alpha_dynamics_B1_253_Arrow_200Hz_AllDays_DaysLabeled_ArtifactCorr xdata ydata labels labels_batch days -v7.3


%% (MAIN) GETTING ALPHA WAVE FOR ARROW DATA, 253 grid COMPLEX



clc;clear
close all

if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')    
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers'))

else
    %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    root_path='/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker';
    cd(root_path)
    %load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/helpers'))

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
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',9, ...
    'SampleRate',1e3);

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7,'HalfPowerFrequency2',9, ...
    'SampleRate',200);


folders={'20240515', '20240517', '20240614', ...
    '20240619', '20240621', '20240626'};



hg_alpha_switch=false; %1 means get hG, 0 means get alpha dynamics

for i=1:length(folders)

    folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
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

    %%%%%% get imagined data files    
    files=[];
    for ii=1:length(imag_idx)
        imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');        
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata);

    else
        [xdata,ydata,idx,trial_idx] = ...
            get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata,1);

    end

    labels = [labels; zeros(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];
    mvmt_labels= [mvmt_labels;trial_idx];

    %%%%%% getting online files now
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    if hg_alpha_switch
        [xdata,ydata,idx,trial_idx] = ...
            get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata,1);

    else
        [xdata,ydata,idx,trial_idx] = ...
            get_spatiotemp_windows(files,d2,ecog_grid,xdata,ydata,1);
    end

    labels = [labels; ones(idx,1)];
    days = [days;ones(idx,1)*i];
    labels_batch = [labels_batch;zeros(idx,1)];
    mvmt_labels= [mvmt_labels;trial_idx];


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
    % if hg_alpha_switch
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


for i=1:length(xdata)
    disp(i/length(xdata)*100)
    tmp=xdata{i};
    tmp = single(tmp);
    xdata{i}=tmp;

    tmp=ydata{i};
    tmp = single(tmp);
    ydata{i}=tmp;
end

%save alpha_dynamics_200Hz_AllDays_zscore xdata ydata labels labels_batch days -v7.3
save alpha_dynamics_B1_253_Arrow_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex xdata ydata labels labels_batch days -v7.3


%% NEW HAND DATA MULTI CYCLIC
% % extract single trials
% % get the time-freq features in raw and in hG
% % train a bi-GRU
% 
% 
% clc;clear
% root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
% foldernames = {'20220608','20220610','20220622','20220624'};
% cd(root_path)
% 
% imagined_files=[];
% for i=1:length(foldernames)
%     folderpath = fullfile(root_path, foldernames{i},'HandImagined')
%     D=dir(folderpath);
% 
%     for j=3:length(D)
%         filepath=fullfile(folderpath,D(j).name,'Imagined');
%         if ~exist(filepath)
%             filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
%         end
%         tmp=dir(filepath);
%         imagined_files = [imagined_files;findfiles('',filepath)'];
%     end
% end
% 
% 
% % load the data for the imagined files, if they belong to right thumb,
% % index, middle, ring, pinky, pinch, tripod, power
% D1i={};
% D2i={};
% D3i={};
% D4i={};
% D5i={};
% D6i={};
% D7i={};
% D8i={};
% D9i={};
% D10i={};
% D11i={};
% D12i={};
% D13i={};
% D14i={};
% for i=1:length(imagined_files)
%     disp(i/length(imagined_files)*100)
%     try
%         load(imagined_files{i})
%         file_loaded = true;
%     catch
%         file_loaded=false;
%         disp(['Could not load ' files{j}]);
%     end
% 
%     if file_loaded
%         action = TrialData.TargetID;
%         %disp(action)
% 
%         % find the bins when state 3 happened and then extract each
%         % individual cycle (2.6s length) as a trial
% 
%         % get times for state 3 from the sample rate of screen refresh
%         time  = TrialData.Time;
%         time = time - time(1);
%         idx = find(TrialData.TaskState==3) ;
%         task_time = time(idx);
% 
%         % get the kinematics and extract times in state 3 when trials
%         % started and ended
%         kin = TrialData.CursorState;
%         kin=kin(1,idx);
%         kind = [0 diff(kin)];
%         aa=find(kind==0);
%         kin_st=[];
%         kin_stp=[];
%         for j=1:length(aa)-1
%             if (aa(j+1)-aa(j))>1
%                 kin_st = [kin_st aa(j)];
%                 kin_stp = [kin_stp aa(j+1)-1];
%             end
%         end
% 
%         %getting start and stop times
%         start_time = task_time(kin_st);
%         stp_time = task_time(kin_stp);
% 
% 
%         % get corresponding neural times indices
%         %         neural_time  = TrialData.NeuralTime;
%         %         neural_time = neural_time-neural_time(1);
%         %         neural_st=[];
%         %         neural_stp=[];
%         %         st_time_neural=[];
%         %         stp_time_neural=[];
%         %         for j=1:length(start_time)
%         %             [aa bb]=min(abs(neural_time-start_time(j)));
%         %             neural_st = [neural_st; bb];
%         %             st_time_neural = [st_time_neural;neural_time(bb)];
%         %             [aa bb]=min(abs(neural_time-stp_time(j)));
%         %             neural_stp = [neural_stp; bb-1];
%         %             stp_time_neural = [stp_time_neural;neural_time(bb)];
%         %         end
% 
%         % get the broadband data for each trial
%         raw_data=cell2mat(TrialData.BroadbandData');
% 
%         % extract the broadband data (Fs-1KhZ) based on rough estimate of
%         % the start and stop times from the kinematic data
%         start_time_neural = round(start_time*1e3);
%         stop_time_neural = round(stp_time*1e3);
%         data_seg={};
%         for j=1:length(start_time_neural)
%             tmp = (raw_data(start_time_neural(j):stop_time_neural(j),:));
%             tmp=tmp(1:round(size(tmp,1)/2),:);
%             % pca step
%             %m=mean(tmp);
%             %[c,s,l]=pca(tmp,'centered','off');
%             %tmp = (s(:,1)*c(:,1)')+m;
%             data_seg = cat(2,data_seg,tmp);
%         end
% 
%         if action==1
%             D1i = cat(2,D1i,data_seg);
%         elseif action==2
%             D2i = cat(2,D2i,data_seg);
%         elseif action==3
%             D3i = cat(2,D3i,data_seg);
%         elseif action==4
%             D4i = cat(2,D4i,data_seg);
%         elseif action==5
%             D5i = cat(2,D5i,data_seg);
%         elseif action==6
%             D6i = cat(2,D6i,data_seg);
%         elseif action==7
%             D7i = cat(2,D7i,data_seg);
%         elseif action==8
%             D8i = cat(2,D8i,data_seg);
%         elseif action==9
%             D9i = cat(2,D9i,data_seg);
%         elseif action==10
%             D10i = cat(2,D10i,data_seg);
%         elseif action==11
%             D11i = cat(2,D11i,data_seg);
%         elseif action==12
%             D12i = cat(2,D12i,data_seg);
%         elseif action==13
%             D13i = cat(2,D13i,data_seg);
%         elseif action==14
%             D14i = cat(2,D14i,data_seg);
%         end
%     end
% end
% 

