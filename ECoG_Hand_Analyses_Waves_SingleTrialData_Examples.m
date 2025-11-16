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

x=[];
for i = 1:length(stats_ol)
    tmp = stats_ol(i).size;
    %tmp = stats_ol(i).corr;
    x(i,:) = smooth(tmp(1:2.2e3),50);
    %tmp = smooth(tmp,50);
    %x= [x; nanmedian(tmp)];
    %x=[x;sum(isnan(tmp))/length(tmp)];
end
figure;plot(mean(x,1))
xol=x;


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
for i = 1:length(stats_cl)
    tmp = stats_cl(i).corr;
    %tmp = smooth(tmp,10);
    %x= [x; nanmedian(tmp)];
    %x(i,:) = smooth(tmp(1:500),50);
    x=[x;sum(isnan(tmp))/length(tmp)];
end
xcl=x;


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

folders={'20240515', '20240517', '20240614', ...
     '20240619', '20240621', '20240626',...
'20240710','20240712','20240731'};
i=1;
folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
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
imag_idx=imag_idx(1);
online_idx=online_idx(1);

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);
d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7.5,'HalfPowerFrequency2',9.5, ...
    'SampleRate',200);
hilbert_flag=1;


%%%%%% get imagined data files
files=[];
for ii=1:length(imag_idx)
    imag_folderpath = fullfile(folderpath, D(imag_idx(ii)).name,'Imagined');
    files = [files;findfiles('mat',imag_folderpath)'];
end
stats_ol = rotational_waves_stats(files,d2,hilbert_flag,ecog_grid);



%%%%%% get batch data files %%%%%
files=[];
for ii=1:length(online_idx)
    imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
    files = [files;findfiles('mat',imag_folderpath)'];
end

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


[p,h] = ranksum(xol,xcl)


