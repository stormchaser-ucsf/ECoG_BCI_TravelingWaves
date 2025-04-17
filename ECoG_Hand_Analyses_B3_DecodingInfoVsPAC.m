
%% (MAIN): LOOKING at the the relationship b/w decoding information at each channel and
% PAC b/w hg and alpha/delta at that channel



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


% d1 = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
%     'SampleRate',1e3);

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',1e3);


d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

d3 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

reg_days=[];
mahab_dist_days=[];
plot_true = false;
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
    %[pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    pac=zeros(100,253);


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
    % y = abs(mean(pac))';
    % x = mahab_dist';
    % x = [ones(size(x,1),1) x];
    % [B,BINT,R,RINT,STATS1] = regress(y,x);
   

    if plot_true

        yhat = x*B;
        plot(x(:,2),yhat,'k')

        % % plot mahab dist on brain
        % plot_on_brain(ch_wts_mahab,cortex,elecmatrix,ecog_grid)
        %   title(['hG decoding info CL Day ' num2str(i)])
        % 
        % % plot PAC on brain
        % plot_on_brain(ch_wts_pac,cortex,elecmatrix,ecog_grid)
        % title(['hG-delta PAC CL Day ' num2str(i)])

    end

    %reg_days(:,i) = [B; STATS1(3)];

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
ylabel('Slope b/w PAC and decoding info.')
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

save mahab_pac_alpha_hg_B3_Hand -v7.3





tmp=angle(mean(pac));
I=find(pac_tmp>0.3);
figure;rose(tmp(I))
title('Preferred angle between hG and alpha')
plot_beautify

