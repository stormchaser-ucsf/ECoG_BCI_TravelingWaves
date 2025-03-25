function [xdata,ydata,idx] = get_spatiotemp_windows_hg(files,d2,ecog_grid,xdata,ydata)

%load the data and extract the downsampled alpha dynamics within a bin and
%the resulting average hG within that same bin


% to make it better, have to extract the windows in a much better way...
% compute hg power at full sampling rate then downsample and average over
% the 200ms bin. 
idx=0;
for ii=1:length(files)
    disp(ii/length(files)*100)
    loaded=1;
    try
        load(files{ii})
    catch
        loaded=0;
    end

    if loaded==1

        % have to take 1s before the start of state3, to 1s after state
        % 4 and then trim



        kinax1 = find(TrialData.TaskState==1);
        kinax2 = find(TrialData.TaskState==2);
        kinax3 = find(TrialData.TaskState==3);
        kinax4 = find(TrialData.TaskState==4);
        kinax3 = [kinax2(3:end) kinax3];

        for xx=1:length(kinax3)
            data = TrialData.BroadbandData{kinax3(xx)};
            data = resample(data,200,1000);
            data = filtfilt(d2,data);
            len = size(data,1);
            if len > 40
                data = data(1:40,:);
            elseif len < 40       
                d = 40-len;
                data(end+1:40,:) = repmat(data(end,:),d,1);
            end
            a=[];
            for j=1:size(data,1)
                aa = data(j,:);
                aa = aa(ecog_grid);
                a(j,:,:) = aa;
            end

            feat = TrialData.SmoothedNeuralFeatures{kinax3(xx)};
            tmpy = feat(1537:1792);
            b = tmpy(ecog_grid);


            xdata = cat(1,(xdata),(a));
            ydata = cat(1,(ydata),(b));
            idx=idx+1;
        end
    end
end
