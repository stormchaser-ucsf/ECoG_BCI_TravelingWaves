function [xdata,ydata,idx] = ...
    get_spatiotemp_windows_hg_alpha(files,d2,ecog_grid,xdata,ydata,d1)

%load the data and extract the features in 8 to 10Hz range, 200ms
%snippets, 201 ms prediction, in 50ms increments
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

        data2 = cell2mat(TrialData.BroadbandData(kinax2)');
        l2 =  length(data2);
        data4 = cell2mat(TrialData.BroadbandData(kinax4)');
        l4 = length(data4);
        data3 = cell2mat(TrialData.BroadbandData(kinax3)');
        l3 = length(data3);

        data = [data2;data3;data4(1:200,:)];
        l22=floor(l2/5); % length of the down sampled signal

        % get hg
        data_hg = abs(hilbert(filtfilt(d1,data)));

        % resample and filter
        data = resample(data,200,1000);
        df = filtfilt(d2,data);
        df = df(l22+1:end,:);

        data_hg = resample(data_hg,200,1000);
        df_hg = filtfilt(d2,data_hg);
        df_hg = df_hg(l22+1:end,:);


        % filter at 1khz
        %df = filtfilt(d1,data);
        for xx=1:39:size(df,1) %39 is the original step forward, 40 if full
            if size(df,1)-xx > 42
                st = xx;
                stp = xx+39; %39
                %yst=xx+1;
                %ystp = xx+1+39;
                tmpx = df(st:stp,:);
                tmpy = df_hg(st:stp,:);
                a=[];
                for j=1:size(tmpx,1)
                    aa = tmpx(j,:);
                    aa = aa(ecog_grid);
                    a(j,:,:) = aa;
                end
                b=[];
                for j=1:size(tmpy,1)
                    bb = tmpy(j,:);
                    bb = bb(ecog_grid);
                    b(j,:,:) = bb;
                end

                
                xdata = cat(1,(xdata),(a));
                ydata = cat(1,(ydata),(b));
                idx=idx+1;
            end
        end
    end
end
