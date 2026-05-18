
%% INIT
clc;clear

if ispc
    addpath('C:\Users\nikic\Documents\MATLAB')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\wave-matlab-master\wave-matlab-master'))
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves')

else

    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))

end

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
load('ECOG_Grid_8596_000067_B3.mat')

%% LOAD THE DATA AND TAKE A LOOK AT TRAJECTORIES

filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/';
folders = {'20250624','20250717'};

files=[];
for i=1:length(folders)
    tmp_f = fullfile(filepath,folders{i},'Robot3D');
    tmp = findfiles('.mat',tmp_f,1)';
    tmp = remove_kf_params(tmp);
    files=[files;tmp];
end

figure;
hold on
xlim([-250 250])
ylim([-250 250])
zlim([-250 250])
targets={};
cols=turbo(6);
for i=1:length(files)
    load(files{i})
    targets(i).tid = TrialData.TargetID;
    targets(i).tpos = TrialData.TargetPosition;
    targets(i).path = TrialData.CursorState;
    targets(i).acc = TrialData.SelectedTargetID == TrialData.TargetID;

    % plot path if accurate
    if targets(i).acc
        cx = TrialData.TargetPosition(1);
        cy = TrialData.TargetPosition(2);
        cz = TrialData.TargetPosition(3);
        width=50;
        plotCube(cx,cy,cz, width);hold on
        plot3(TrialData.CursorState(1,:),TrialData.CursorState(2,:),...
            TrialData.CursorState(3,:),'Color',cols(TrialData.TargetID,:),...
            'LineWidth',3)
    end
end

% to see which colors are there
figure;
for i = 1:size(cols,1)
    rectangle('Position', [i 0 1 1], ...
        'FaceColor', cols(i,:), ...
        'EdgeColor', 'k');
    text(i+0.5, -0.2, num2str(i), ...
        'HorizontalAlignment', 'center');
end
axis equal
xlim([1 7])
ylim([-0.5 1])
axis off

%% LOAD THE FILES AND LOOK AT MU WAVE STABILITY
% the mu stuff runs at 50hz.
% gate decoding if there is a mu wave event

d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',7.0,'HalfPowerFrequency2',10, ...
    'SampleRate',50);


% load the data
for i=1:length(files)

    if targets(i).acc==1

        load(files{i})

        kinax1 = find(TrialData.TaskState==1);
        kinax2 = find(TrialData.TaskState==2);
        kinax3 = find(TrialData.TaskState==3);
        kinax4 = find(TrialData.TaskState==4);

        %tmp=cell2mat(TrialData.BroadbandData);

        data2 = cell2mat(TrialData.BroadbandData(kinax2)');
        l2 =  length(data2);
        data4 = cell2mat(TrialData.BroadbandData(kinax4)');
        l4 = length(data4);
        data3 = cell2mat(TrialData.BroadbandData(kinax3)');
        l3 = length(data3);
        data1 = cell2mat(TrialData.BroadbandData(kinax1)');
        l1 = length(data1);

        data = [data1;data2;data3;data4];
        data_main=data;
        tmain = 1:size(data,1); % in ms the true time

        ds_fac=1e3/d2.SampleRate;
        l22=floor(l2/ds_fac); % length of the down sampled signal
        l11=floor(l1/ds_fac); % length of the down sampled signal
        l33=floor(l3/ds_fac); % length of the down sampled signal
        l44=floor(l4/ds_fac); % length of the down sampled signal

        % filter in mu band to get mu signal
        data = resample(data,d2.SampleRate,1000);
        df = filtfilt(d2,data);
        df= hilbert(df);

        % extract only in state 3
        st = l11+l22+1;
        stp = st+l33-1;
        df = df(st:stp,:);

        % keep track of time
        tcut = tmain(st:stp); % what is being taken
        tcut = tcut(1:20:end);% down sampled to 50Hz


        % extract wave stability metric
        % detect planar waves across mini-grid location
        planar_val_time=[];
        smooth1_vals=[];
        smooth2_vals=[];
        parfor t=1:size(df,1)
            %disp(t)

            % estimate planar waves across mini grid
            tmp = df(t,:);
            xph = tmp(ecog_grid);

            [planar_val,aa,bb] = planar_stats_muller(xph);
            smooth1_vals(t)=aa;
            smooth2_vals(t)=bb;
            planar_val_time(t,:,:) = planar_val;
        end

        %%%% phase gradient stability
        stab=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));
        end

        %decodes = TrialData.FilteredClickerState;
        decodes = TrialData.ClickerState;
        time_decodes = (0:length(decodes)-1) * (1/5);
        time_waves = (0:length(stab)-1) * (1/50);
        figure;hold on
        plot(time_waves,zscore(stab))
        plot(time_decodes,decodes==TrialData.TargetID)
        hline(0,'--k')
    end

end

