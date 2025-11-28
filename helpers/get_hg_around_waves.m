function hg_act =  get_hg_around_waves(files,d3,ecog_grid,r,c)


hg_act={};
parfor ii=1:length(files)
    disp(['Processing file ' num2str(ii) ' of ' num2str(length(files))])
    loaded=1;
    try
        out=load(files{ii});
    catch
        loaded=0;
        disp(['not loaded file  ' files{ii}])
    end

    if loaded==1

        TrialData = out.TrialData;
        kinax1 = find(TrialData.TaskState==1);
        kinax2 = find(TrialData.TaskState==2);
        kinax3 = find(TrialData.TaskState==3);
        kinax4 = find(TrialData.TaskState==4);

        data2 = cell2mat(TrialData.BroadbandData(kinax2)');
        l2 =  length(data2);
        data4 = cell2mat(TrialData.BroadbandData(kinax4)');
        l4 = length(data4);
        data3 = cell2mat(TrialData.BroadbandData(kinax3)');
        l3 = length(data3);
        data1 = cell2mat(TrialData.BroadbandData(kinax1)');
        l1 = length(data1);

        %data = [data1;data2;data3;data4];
        data = [data1;data2;data3];
        l22=floor(l2/5); % length of the down sampled signal
        l11=floor(l1/5); % length of the down sampled signal

        % get the hG signal downsampled to 200Hz
        data_hg = filtfilt(d3,data);
        data_hg = abs(hilbert(data_hg));
        data_hg = resample(data_hg,200,1000);
        data_hg = data_hg(l11+1:end,:);

        % get the pc of the activation in the minigrid region
        mini_grid = ecog_grid(r,c);
        data_hg = data_hg(:,mini_grid);
        s = mean(data_hg,2);
        %[c1,s,l] = pca(data_hg);

        % store
        hg_act{ii} = s(:,1);
    end
end


end