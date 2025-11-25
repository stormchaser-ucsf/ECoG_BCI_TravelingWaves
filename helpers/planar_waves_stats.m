function stats = planar_waves_stats(files,d2,hilbert_flag,ecog_grid)


stats={};
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

        % resample and filter
        data = resample(data,200,1000);
        df = filtfilt(d2,data);

        % get the hilbert transform of the signal
        if hilbert_flag
            df= hilbert(df);
        end
        df = df(l11+1:end,:);

        % detect planar waves across mini-grid location

        planar_val_time={};
        parfor t=1:size(df,1)
            %disp(t)
            
            tmp = df(t,:);
            xph = tmp(ecog_grid);

            % smooth phase
            % [XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );
            % M = real(xph);
            % N = imag(xph);
            % M = smoothn(M,'robust'); %dx
            % N = smoothn(N,'robust'); %dy
            % xphs = M + 1j*N; % smoothed phasor field: gets smoothed estimates of phase


            % estimate planar waves across mini grid
            planar_val = planar_stats(xph);
            planar_val_time{t} = planar_val;
        end
    end
    b=cell2mat(planar_val_time');
    r1=[b(1:end).rho];    
    stats(ii).corr = r1;
end


end


