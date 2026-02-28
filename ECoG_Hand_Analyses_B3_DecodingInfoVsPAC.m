
%% (MAIN): LOOKING at the the relationship b/w decoding information at each channel and
% PAC b/w hg and alpha/delta at that channel
% B3



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
    %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
end


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

% d1 = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
%     'SampleRate',1e3);


d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

d3 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

reg_days=[];
mahab_dist_days=[];
plot_true = true;
close all

imaging_B3_waves;
close all


for i=1:length(session_data)



    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    %batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    %batch_idx = find(folders_batch==1);
    %batch_idx1 = find(folders_batch1==1);

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    if sum(folders_batch1)==0
        batch_idx = find(folders_batch==1);
    else
        batch_idx = find(folders_batch1==1);
    end




    %     if sum(folders_batch1)==0
    %         batch_idx = find(folders_batch==1);
    %     else
    %         batch_idx = find(folders_batch1==1);
    %     end
    %     %batch_idx = [online_idx batch_idx];



    online_idx=[online_idx batch_idx];
    %batch_idx = [online_idx batch_idx_overall];


    % %%%%% get imagined data files
    % folders = session_data(i).folders(imag_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    %
    % % get the phase locking value
    % disp(['Processing Day ' num2str(i) ' OL'])
    % [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %
    % % get the mahab dist at each channel
    % [mahab_dist] = get_mahab_dist(files);
    %
    % % plot and see?
    % figure;
    % plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
    % xlabel('Mahab Dist')
    % ylabel('PAC')


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
    %pac=zeros(100,253);


    % get the mahab dist at each channel
    [mahab_dist] = get_mahab_dist(files);
    mahab_dist_days(i,:) = mahab_dist;


    if plot_true

        figure;
        % plot pac as brain image
        pac_tmp = abs(mean(pac));
        ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
            pac_tmp(116:end)];
        %figure;
        subplot(1,3,1)
        imagesc(ch_wts(ecog_grid))
        title(['PAC Day ' num2str(i)])
        ch_wts_pac=ch_wts;        

        % plot mahab dist as brain image
        ch_wts = [mahab_dist(1:107) 0 mahab_dist(108:111) 0  mahab_dist(112:115) 0 ...
            mahab_dist(116:end)];
        %figure;
        subplot(1,3,2)
        imagesc(ch_wts(ecog_grid))
        title(['Mahab dist Day ' num2str(i)])
        ch_wts_mahab=ch_wts;

        % hand knob weights
        %hnd2 = ch_wts(ecog_grid(2:3,3:4));


        % plot and see?
        %figure;
        subplot(1,3,3)
        plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
        xlabel('Mahab Dist')
        ylabel('PAC')
        title(['CL Day ' num2str(i)])
        ylim([0 0.7])
        plot_beautify
        hold on

    end




    % regression
    y = abs(mean(pac))';
    x = mahab_dist';
    x = [ones(size(x,1),1) x];
    [B,BINT,R,RINT,STATS1] = regress(y,x);
   

    if plot_true

        yhat = x*B;
        plot(x(:,2),yhat,'k','LineWidth',1)

        % plot mahab dist on brain
        figure;
        plot_on_brain(ch_wts_mahab,cortex,elecmatrix,ecog_grid)
          title(['hG decoding info CL Day ' num2str(i)])
        % 
        % % plot PAC on brain
        % %figure
        % plot_on_brain(ch_wts_pac,cortex,elecmatrix,ecog_grid)
        % title(['hG-delta PAC CL Day ' num2str(i)])

    end

    reg_days(:,i) = [B; STATS1(3)];

    %
    % %%%%%%%%% getting batch files now
    % folders = session_data(i).folders(batch_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    %
    % % get the phase locking value
    % disp(['Processing Day ' num2str(i) ' CL'])
    % [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %
    %
    % % get the mahab dist at each channel
    % [mahab_dist] = get_mahab_dist(files);
    %
    %
    % % plot pac on brain
    % pac_tmp = abs(mean(pac));
    % ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
    %     pac_tmp(116:end)];
    % figure;
    % imagesc(ch_wts(ecog_grid))
    %
    % % plot mahab dist on brain grid
    % ch_wts = [mahab_dist(1:107) 0 mahab_dist(108:111) 0  mahab_dist(112:115) 0 ...
    %     mahab_dist(116:end)];
    % figure;
    % imagesc(ch_wts(ecog_grid))
    %
    % % hand knob weights
    % hnd9b = ch_wts(ecog_grid(2:3,3:4));
    %
    %
    % % plot and see?
    % figure;
    % plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
    % xlabel('Mahab Dist')
    % ylabel('PAC')
    % title(['CL Day ' num2str(i)])
    % ylim([0 0.7])
    % plot_beautify
    %


end

figure;hold on
plot(log(reg_days(3,:)),'.','MarkerSize',20)
xlabel('Days')
ylabel('Log P-value')
xticks(1:10)
y = log(reg_days(3,:))';
x = (1:10)';
x = [ones(size(x,1),1) x];
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
%plot(x(:,2),yhat,'k')
plot_beautify
title('Evolution of alpha-hG PAC and discriminability (sig)')
hline(log(0.05),'--r')



figure;hold on
plot((reg_days(2,:)),'.','MarkerSize',20)
xlabel('Days')
%ylabel('Slope b/w PAC and decoding info.')
ylabel('LFO - hG PAC and hG decoding info.')
xticks(1:10)
y = reg_days(2,:)';
x = (1:10)';
x = [ones(size(x,1),1) x];
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
plot(x(:,2),yhat,'k')
plot_beautify
title('Evolution of alpha-hG PAC and discriminability')
xlim([0.5 10.5])

save mahab_pac_alpha_hg_B3_Hand_New -v7.3


tmp=angle(mean(pac));
I=find(pac_tmp>0.3);
figure;rose(tmp(I))
title('Preferred angle between hG and alpha')
plot_beautify


% % plot mahab distances days
for i=1:size(mahab_dist_days,1)

    mahab_dist_day1 = mahab_dist_days(i,:);
    ch_wts1 = [mahab_dist_day1(1:107) 0 mahab_dist_day1(108:111) 0  mahab_dist_day1(112:115) 0 ...
        mahab_dist_day1(116:end)];

    plot_on_brain(ch_wts1,cortex,elecmatrix,ecog_grid)
    title(['hG decoding info CL Day ' num2str(i)])

end

%% (B1 ARROW 253 GRID): LOOKING at the the relationship b/w decoding 
% information at each channel and % PAC b/w hg and alpha/delta at that 
% channel





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
    imaging_B1_253;close all

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves/helpers'))

end


% d1 = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
%     'SampleRate',1e3);

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',1e3); % or 7 to 9 for b1 253 grid


d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

d3 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',1e3);


folders={'20240515', '20240517', '20240614', '20240619', '20240621', '20240626'};



reg_days=[];
mahab_dist_days=[];
plot_true = true;
pac_days=[];

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



    % %%%%% get imagined data files
    % folders = session_data(i).folders(imag_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    %
    % % get the phase locking value
    % disp(['Processing Day ' num2str(i) ' OL'])
    % [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %
    % % get the mahab dist at each channel
    % [mahab_dist] = get_mahab_dist(files);
    %
    % % plot and see?
    % figure;
    % plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
    % xlabel('Mahab Dist')
    % ylabel('PAC')


    %%%%%% getting online files now     
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' CL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %pac=zeros(100,253);
    pac_days(i,:)=abs(mean(pac));


    % get the mahab dist at each channel
    [mahab_dist] = get_mahab_dist_7DoF(files);
    mahab_dist_days(i,:) = mahab_dist;


    if plot_true

        figure;
        % plot pac as brain image
        pac_tmp = abs(mean(pac));
        ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
            pac_tmp(116:end)];
        %figure;
        subplot(1,3,1)
        imagesc(ch_wts(ecog_grid))
        title(['PAC Day ' num2str(i)])
        ch_wts_pac=ch_wts;        

        % plot mahab dist as brain image
        ch_wts = [mahab_dist(1:107) 0 mahab_dist(108:111) 0  mahab_dist(112:115) 0 ...
            mahab_dist(116:end)];
        %figure;
        subplot(1,3,2)
        imagesc(ch_wts(ecog_grid))
        title(['Mahab dist Day ' num2str(i)])
        ch_wts_mahab=ch_wts;

        % hand knob weights
        %hnd2 = ch_wts(ecog_grid(2:3,3:4));


        % plot and see?
        %figure;
        subplot(1,3,3)
        plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
        xlabel('Mahab Dist')
        ylabel('PAC')
        title(['CL Day ' num2str(i)])
        ylim([0 0.7])
        plot_beautify
        hold on

    end




    % regression
    y = abs(mean(pac))';
    x = mahab_dist';
    x = [ones(size(x,1),1) x];
    [B,BINT,R,RINT,STATS1] = regress(y,x);
   

    if plot_true

        yhat = x*B;
        plot(x(:,2),yhat,'k')

        % plot mahab dist on brain
        ch_wts_mahab=mahab_dist;
        plot_on_brain_B1_253(ch_wts_mahab,cortex,elecmatrix,ecog_grid,15)
          title(['hG decoding info CL Day ' num2str(i)])

        % plot PAC on brain
        ch_wts_pac = abs(mean(pac));
        plot_on_brain_B1_253(ch_wts_pac,cortex,elecmatrix,ecog_grid,25)
        title(['hG-delta PAC CL Day ' num2str(i)])

    end

    reg_days(:,i) = [B; STATS1(3)];

    %
    % %%%%%%%%% getting batch files now
    % folders = session_data(i).folders(batch_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    %
    % % get the phase locking value
    % disp(['Processing Day ' num2str(i) ' CL'])
    % [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %
    %
    % % get the mahab dist at each channel
    % [mahab_dist] = get_mahab_dist(files);
    %
    %
    % % plot pac on brain
    % pac_tmp = abs(mean(pac));
    % ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
    %     pac_tmp(116:end)];
    % figure;
    % imagesc(ch_wts(ecog_grid))
    %
    % % plot mahab dist on brain grid
    % ch_wts = [mahab_dist(1:107) 0 mahab_dist(108:111) 0  mahab_dist(112:115) 0 ...
    %     mahab_dist(116:end)];
    % figure;
    % imagesc(ch_wts(ecog_grid))
    %
    % % hand knob weights
    % hnd9b = ch_wts(ecog_grid(2:3,3:4));
    %
    %
    % % plot and see?
    % figure;
    % plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
    % xlabel('Mahab Dist')
    % ylabel('PAC')
    % title(['CL Day ' num2str(i)])
    % ylim([0 0.7])
    % plot_beautify
    %


end

% plotting 

figure;hold on
plot(log(reg_days(3,:)),'.','MarkerSize',20)
xlabel('Days')
ylabel('Log P-value')
xticks(1:size(reg_days,2))
y = log(reg_days(3,:))';
x = (1:size(reg_days,2))';
x = [ones(size(x,1),1) x];
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
%plot(x(:,2),yhat,'k')
plot_beautify
title('Evolution of alpha-hG PAC and discriminability (sig)')
hline(log(0.05),'--r')



figure;hold on
plot((reg_days(2,:)),'.','MarkerSize',20)
xlabel('Days')
ylabel('Slope b/w PAC and decoding info.')
xticks(1:size(reg_days,2))
y = reg_days(2,:)';
x = (1:size(reg_days,2))';
x = [ones(size(x,1),1) x];
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
plot(x(:,2),yhat,'k')
plot_beautify
title('Evolution of delta-hG PAC and discriminability')
xlim([0.5 6.5])


% tmp=angle(mean(pac));
% I=find(pac_tmp>0.3);
% figure;rose(tmp(I))
% title('Preferred angle between hG and alpha')
% plot_beautify



save mahab_pac_delta_hg_B1_253_7DoF -v7.3


