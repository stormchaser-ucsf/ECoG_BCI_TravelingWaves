function [data_trial] = ...
    get_spatiotemp_windows_trial(files,d2,ecog_grid,data_trial,bci_type,day_idx)
%function [data] = get_spatiotemp_windows_trial(files,d2,ecog_grid,data,bci_type)

%load the data and extract the features in 8 to 10Hz range, 200ms
%snippets, 201 ms prediction, in 50ms increments
idx=0;trial_idx=[];
for ii=1:length(files)
    disp(ii/length(files)*100)
    loaded=1;
    try
        load(files{ii})
    catch
        loaded=0;
    end

    if loaded==1

        xdata={};ydata={};

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
        data1 = cell2mat(TrialData.BroadbandData(kinax1)');
        l1 = length(data1);

        data = [data1;data2;data3;data4];

        % filter at 1Khz
        df = filtfilt(d2,data);

        % extract only state 2 onwards
        df =df(l1+1:end,:);

        % downsample to 100Hz
        df = resample(df,100,1000);
        %
        %
        %
        %
        %
        %         l22=floor(l2/5); % length of the down sampled signal
        %         l11=floor(l1/5); % length of the down sampled signal
        %
        %         % resample and filter
        %         data = resample(data,200,1000);
        %         df = filtfilt(d2,data);
        %         df = df(l11+1:end,:);

        %         % filter at 1khz
        %         %df = filtfilt(d1,data);


        % keep track of trial segments
        trial_seg=0;

        % getting 200ms segments
        %         for xx=1:39:length(df) %39 is the original step forward
        %             if size(df,1)-xx > 42
        %                 st = xx;
        %                 stp = xx+39;
        %                 yst=xx+1;
        %                 ystp = xx+1+39;
        %                 tmpx = df(st:stp,:);
        %                 tmpy = df(yst:ystp,:);
        %                 a=[];
        %                 for j=1:size(tmpx,1)
        %                     aa = tmpx(j,:);
        %                     aa = aa(ecog_grid);
        %                     a(j,:,:) = aa;
        %                 end
        %                 b=[];
        %                 for j=1:size(tmpy,1)
        %                     bb = tmpy(j,:);
        %                     bb = bb(ecog_grid);
        %                     b(j,:,:) = bb;
        %                 end
        %
        %
        %                 xdata = cat(1,(xdata),(a));
        %                 ydata = cat(1,(ydata),(b));
        %                 idx=idx+1;
        %                 trial_seg = trial_seg+1;
        %             end
        %         end

        % getting 1s segments with 200ms overlap
        for xx=1:25:length(df) %39 is the original step forward
            if size(df,1)-xx >= 100
                st = xx;
                stp = xx+99;
                %yst=xx+1;
                %ystp = xx+1+39;
                tmpx = df(st:stp,:);
                %tmpy = df(yst:ystp,:);
                a=[];
                for j=1:size(tmpx,1)
                    aa = tmpx(j,:);
                    aa = aa(ecog_grid);
                    a(j,:,:) = aa;
                end
                %                  b=[];
                %                  for j=1:size(tmpy,1)
                %                      bb = tmpy(j,:);
                %                      bb = bb(ecog_grid);
                %                      b(j,:,:) = bb;
                %                  end


                xdata = cat(1,(xdata),(a));
                %ydata = cat(1,(ydata),(b));
                idx=idx+1;
                trial_seg = trial_seg+1;
            end
        end




    end

    if length(data_trial)==0
        len = 1;
    else
        len = length(data_trial)+1;
    end


    data_trial(len).xdata=xdata;
    %data_trial(len).ydata=ydata;
    data_trial(len).TargetID = TrialData.TargetID;

    if bci_type==0
        data_trial(len).BCI_context = 'OL';
    elseif    bci_type==1
        data_trial(len).BCI_context = 'CL';
    end

    data_trial(len).Day=day_idx;


    %trial_idx = [trial_idx ;[TrialData.TargetID * ones(trial_seg,1)]];
end


