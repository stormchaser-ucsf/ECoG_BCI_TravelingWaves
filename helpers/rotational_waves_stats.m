function [stats] = rotational_waves_stats(files,d2,hilbert_flag,ecog_grid)
%function [stats] = rotational_waves_stats(files,d2,hilbert_flag,ecog_grid)


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
        %df = df(l11+1:end,:);

        % detect rotations per time - point

        rot_size=[];corr_val_time=[];
        parfor t=1:size(df,1)
            vortices={};
            tmp = df(t,:);
            xph = tmp(ecog_grid);

            % smooth phase and then estimate gradient and then curl
            [curl_val,M,N,XX,YY,xphs] = get_curl(xph);

            % if curl passes threshold, send to get circular circular
            % correlation between rotation angle and signal phase (data
            % driven extents)
            if max((curl_val(:))) > 0.75
                xph=xph;
                vortices = ...
                    detect_curlVortices_dataDriven_any(curl_val, xph,xphs,0.75);
            end
            I = zeros(11,23);corr_val=NaN(length(vortices),1);
            for k=1:length(vortices)
                r = vortices(k).rows;
                c = vortices(k).cols;
                if ~isempty(r) && ~isempty(c)
                    I(r(1):r(2),c(1):c(2))=I(r(1):r(2),c(1):c(2)) + 1;
                    corr_val(k) = vortices(k).corr;
                end
            end
            I(I>0)=1;
            rot_size(t) = sum((I(:))) / numel(I);
            corr_val_time(t) = nanmean(corr_val);
        end
    end

    stats(ii).size = rot_size;
    stats(ii).corr = corr_val_time;
end

end