function [xdata,ydata,idx,trial_idx] = ...
    get_spatiotemp_windows_roi(files,d2,ecog_grid,xdata,ydata,rows,cols,hilbert_flag)


if nargin<8
    hilbert_flag=false;
    disp('Extracting only real data')
else
    hilbert_flag=true;
    disp('Extracting complex data')
end


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
        disp(['not loaded file  ' files{ii}])
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
        data1 = cell2mat(TrialData.BroadbandData(kinax1)');
        l1 = length(data1);

        data = [data1;data2;data3;data4];
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

        

        % keep track of trial segments
        trial_seg=0;


        % filter at 1khz
        %df = filtfilt(d1,data);
        for xx=1:39:length(df) %39 is the original step forward
            if size(df,1)-xx > 42
                st = xx;
                stp = xx+39;
                yst=xx+1;
                ystp = xx+1+39;
                tmpx = df(st:stp,:);
                tmpy = df(yst:ystp,:);
                a=[];
                for j=1:size(tmpx,1)
                    
                    aa = tmpx(j,:);

                    % Artifact correction
                    idx1=find(abs(aa)>5);
                    if hilbert_flag
                         aa(idx1) = (randn(size(idx1)) + 1i*randn(size(idx1)))*1e-5;
                    else
                        aa(idx1) = randn(size(idx1))*1e-5;
                    end

                    aa = aa(ecog_grid(rows,cols));
                    a(j,:,:) = aa;
                end
                b=[];
                for j=1:size(tmpy,1)
                    bb = tmpy(j,:);
                    
                      % Artifact correction
                    idx1=find(abs(bb)>5);
                    if hilbert_flag
                        bb(idx1) = (randn(size(idx1)) + 1i*randn(size(idx1)))*1e-5;
                    else
                        bb(idx1) = randn(size(idx1))*1e-5;
                    end

                    bb = bb(ecog_grid(rows,cols));
                    b(j,:,:) = bb;
                end

                
                xdata = cat(1,(xdata),single(a));
                ydata = cat(1,(ydata),single(b));
                idx=idx+1;
                trial_seg = trial_seg+1;
                if max(abs(a(:)))>5
                    disp(length(xdata));
                end
            end
        end
        trial_idx = [trial_idx ;[TrialData.TargetID * ones(trial_seg,1)]];
    end
    
end


