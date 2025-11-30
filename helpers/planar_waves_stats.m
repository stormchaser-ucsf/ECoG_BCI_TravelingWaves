function stats = planar_waves_stats(files,d2,...
    hilbert_flag,ecog_grid,grid_layout,elecmatrix)


stats={};kk=1;
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

        data = [data1;data2;data3;data4];
        %data = [data1;data2;data3];
        ds_fac=1e3/d2.SampleRate;
        l22=floor(l2/ds_fac); % length of the down sampled signal
        l11=floor(l1/ds_fac); % length of the down sampled signal

        % get the hG signal downsampled to 200Hz
        % data_hg = filtfilt(d3,data);
        % data_hg = abs(hilbert(data_hg));
        % data_hg = resample(data_hg,200,1000);        

        % resample and filter in mu band
        data = resample(data,d2.SampleRate,1000);
        df = filtfilt(d2,data);

        % get the hilbert transform of the signal
        if hilbert_flag
            df= hilbert(df);
        end
        df = df(l11+1:end-40,:);%remove last 800ms

        

        % detect planar waves across mini-grid location

        planar_val_time=[];
        parfor t=1:size(df,1)
            %disp(t)
            
            % estimate planar waves across mini grid
            tmp = df(t,:);
            xph = tmp(ecog_grid);
            
            % smooth phase
            % [XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );
            % M = real(xph);
            % N = imag(xph);
            % M = smoothn(M,'robust'); %dx
            % N = smoothn(N,'robust'); %dy
            % xph = M + 1j*N; % smoothed phasor field: gets smoothed estimates of phase            
            
            %planar_val = planar_stats(xph);            

            % estimate planar waves across the entire grid, fitting wave
            % patterns at a local subcluster around each electrode
            planar_val = planar_stats_full(xph,grid_layout,elecmatrix);            
            %planar_val_time{t} = planar_val;
            planar_val_time(t,:,:) = planar_val;
        end

        %%%% if performing local circular linear correlation around entire grid
        stab=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));
        end

        stats(kk).stab = stab;kk=kk+1;
    end

    %%%%% if just using one grid
    % b=cell2mat(planar_val_time');
    % r1=[b(1:end).rho];    
    % alp=[b(1:end).alp];    
    % stats(ii).corr = r1;    
    % stab=[];
    % for k=2:length(r1)
    %     tmp = cosd(alp(k)) + 1i*sind(alp(k));
    %     tmpm1 = cosd(alp(k-1)) + 1i*sind(alp(k-1));
    %     stab(k) = abs((r1(k)*tmp) - (r1(k-1)*tmpm1));
    % end
    %figure;plot(zscore(stab))

   

end


end


