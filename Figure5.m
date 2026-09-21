% goal here is to show examples of traveling waves


% go through a day's data and get the trial with longest median wave
% duration
% B3, load the prelim from ECoG_Waves_PreProcessSubjectData.m

days=6;
cl_chk=1;
folders_imag =  strcmp(session_data(days).folder_type,'I');
folders_online = strcmp(session_data(days).folder_type,'O');
folders_batch = strcmp(session_data(days).folder_type,'B');
folders_batch1 = strcmp(session_data(days).folder_type,'B1');
imag_idx = find(folders_imag==1);
online_idx = find(folders_online==1);
batch_idx = find(folders_batch==1);
batch_idx1 = find(folders_batch1==1);
online_idx=[online_idx batch_idx batch_idx1];

folders = session_data(days).folders(online_idx);
day_date = session_data(days).Day;
files=[];
for ii=1:length(folders)
    %folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
    %cd(folderpath)
    files = [files;findfiles('mat',folderpath)'];
end


wav_dur=[];
num_waves=[];
good_ch=ones(256,1);
good_ch([108 113 118])=0;
good_ch = logical(good_ch);
vec_field={};
stats={};kk=1;
stats_hg={};
for ii=1:length(files)

    disp(['Processing file ' num2str(ii) ' of ' num2str(length(files))])
    loaded=1;
    try
        load(files{ii})
    catch
        loaded=0;
        disp(['not loaded file  ' files{ii}])
    end

    if loaded==1

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

        % constructing the output vector
        output = NaN(size(data,1),1);
        if cl_chk==1
            t3start = l1+l2+1;
            t3end = t3start + l3;
            k=1;
            for t=t3start:200:(t3end-200)
                if k>length(TrialData.ClickerState)
                    break
                end
                output(t:t+199) = TrialData.ClickerState(k);
                k=k+1;
            end
        end


        %data = [data1;data2;data3];
        ds_fac=1e3/d2.SampleRate;
        l22=floor(l2/ds_fac); % length of the down sampled signal
        l11=floor(l1/ds_fac); % length of the down sampled signal

        % get the hG envelope
        % hg = filtfilt(bpFilt,data);
        % hg = abs(hilbert(hg));

        % get hg through filter bank approach
        Params=TrialData.Params;
        filtered_data=[];
        for k=9:16
            tmp = filtfilt(Params.FilterBank(k).b, ...
                Params.FilterBank(k).a, ...
                data);
            tmp=abs(hilbert(tmp));
            filtered_data = cat(3,filtered_data,tmp);
        end
        hg = squeeze(mean(filtered_data,3));

        % downsample to 50Hz
        hg = resample(hg,d2.SampleRate,1e3);

        % smooth it -> dont smooth for delta PAC
        % hg_smooth=[];
        % for j=1:size(hg,2)
        %     hg_smooth(:,j) = smooth(hg(:,j),10);
        % end
        % hg=hg_smooth;

        % get the mu signal phase of hG
        hg_mu = filtfilt(d2,hg);
        hg_mu = (hilbert(hg_mu));


        % filter in mu band to get mu signal
        data = resample(data,d2.SampleRate,1000);
        df = filtfilt(d2,data);

        %downsample output vector to 50hz
        output = output(1:20:end);

        % get the hilbert transform of the mu signal
        if hilbert_flag
            df= hilbert(df);
        end

        % get delta signal
        ds = filtfilt(bpFilt,data);
        ds = hilbert(ds);

        % get the delta signal phase of hG
        hg_delta = filtfilt(bpFilt,hg);
        hg_delta = (hilbert(hg_delta));

        % remove non-task periods
        df = df(l11+1:end-40,:);%remove last 800ms for b1,b6, last 1000ms for b3 hand
        hg = hg(l11+1:end-40,:);%remove last 800ms for b1,b6, last 1000ms for b3 hand
        hg_mu = hg_mu(l11+1:end-40,:);
        ds = ds(l11+1:end-40,:);
        hg_delta = hg_delta(l11+1:end-40,:);
        output = output(l11+1:end-40);
        output(isnan(output)) = 1e-6;
        output(output~=TrialData.TargetID)=0;
        output(output==TrialData.TargetID)=1;

        % keep track of time
        tcut = tmain(l1:end-800); % what is being taken
        tcut = tcut(1:20:end);% down sampled to 50Hz

        % detect planar waves across mini-grid location
        planar_val_time=[];planar_val_time_hg=[];
        planar_val_time_local=[];
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

            % doing it local over M1
            planar_val_time_local(t,:,:) = planar_val(3:8,1:5);

            % wave detection for hg mu signal
            % tmp = hg_mu(t,:);
            % xph = tmp(ecog_grid);
            % planar_val = planar_stats_muller(xph);
            % planar_val_time_hg(t,:,:) = planar_val;
        end

        %%%% if performing local circular linear correlation around entire grid
        stab=[];stab_hg=[];stab_local=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));

            xt = planar_val_time_local(k,:,:);xt=xt(:);
            xtm1 = planar_val_time_local(k-1,:,:);xtm1=xtm1(:);
            stab_local(k-1) = - mean(abs(xt - xtm1));

            % xt = planar_val_time_hg(k,:,:);xt=xt(:);
            % xtm1 = planar_val_time_hg(k-1,:,:);xtm1=xtm1(:);
            % stab_hg(k-1) = - mean(abs(xt - xtm1));
        end

        % figure;plot(zscore(stab))
        % hline(0)
        % hold on
        % plot(zscore(stab_local))

        stats(kk).stab = stab;
        stats(kk).vec_field = planar_val_time;
        stats(kk).target_id = TrialData.TargetID;

        %%%%% SAVE TRIAL PERFORMANCE
        if cl_chk==1
            click_state = TrialData.FilteredClickerState(TrialData.FilteredClickerState>0);
            if mode(click_state) == TrialData.TargetID
                stats(kk).accuracy=1;
            else
                stats(kk).accuracy=0;
            end
            stats(kk).output=output;
        else
            stats(kk).output=0;
            stats(kk).accuracy=NaN;
        end

        %%%%% STABILITY AND WAVE DETECTION
        % look 300 after start of state 2
        stab1 = zscore(stab(15:end));
        [out,st,stp] = wave_stability_detect(stab1);
        st = st+14;
        stp = stp+14;
        wav_dur(ii) = median(out);
        num_waves(ii) = length(out);
    end

end

wave_dur=wav_dur;
figure;stem(wave_dur)
figure;plot(wave_dur,num_waves,'.','MarkerSize',20)
xlabel('Wave dur')
ylabel('Num waves')

idx=(num_waves==5) .* (wave_dur>=10);
aa=find(idx==1)


% 29th trial, day = 6;
ii=29;
[stab,stab1]=compute_one_file_waves(files,ii,...
     d2,hilbert_flag,ecog_grid,grid_layout,elecmatrix,bpFilt,d1);
figure;plot(zscore(stab));hline(0)
title(num2str(ii))



stab1 = zscore(stab(15:end));
[out,st,stp] = wave_stability_detect(zscore(stab));
out = out(3:end);
st = st(3:end);
stp = stp(3:end);
%st = st+14;
%stp = stp+14;
tt = (1e3/50)*(0:(length(stab)-1));
figure;plot(tt,zscore(stab),'k','LineWidth',1);hline(0)
col = jet(length(out));
col(3,:) = [1 0 1];
col(2,:) =[0 1 0];
for i=1:length(out)
    h=vline(tt(st(i)),'--');
    h.Color = col(i,:);   
    h.LineWidth = 1;

    h=vline(tt(stp(i)),'--');
    h.Color = col(i,:);    
    h.LineWidth = 1;
end
xlabel('Time (ms)')
ylabel('Wave Stability (z)')
plot_beautify
xlim([500 2500])
ylim([-3.1 3.1])

