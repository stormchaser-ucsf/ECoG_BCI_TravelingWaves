function [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2)


%load the data and extract PAC at each channel
idx=0;
pac=[];
hg_alpha_phase={};
alpha_phase={};
bad_ch=[108 113 118];
good_ch=ones(256,1);
good_ch(bad_ch)=0;

for ii=1:length(files)
    %disp(ii/length(files)*100)
    loaded=1;
    try
        load(files{ii})
    catch
        loaded=0;
    end

    if loaded==1

        kinax1 = find(TrialData.TaskState==1);
        kinax2 = find(TrialData.TaskState==2);
        kinax3 = find(TrialData.TaskState==3);
        kinax4 = find(TrialData.TaskState==4);

        data1 = cell2mat(TrialData.BroadbandData(kinax1)');
        l1 =  size(data1,1);
        data2 = cell2mat(TrialData.BroadbandData(kinax2)');
        l2 =  size(data2,1);
        data4 = cell2mat(TrialData.BroadbandData(kinax4)');
        l4 = size(data4,1);
        data3 = cell2mat(TrialData.BroadbandData(kinax3)');
        l3 = size(data3,1);

        data = [data2;data3];
        %data = [data1;data2]; % only state 1 and 2

        % extract hG envelope signal
        hg = filtfilt(d2,data);
        hg = abs(hilbert(hg));

        % extract the alpha component of the hg envelope
        hg_alpha = filtfilt(d1,hg);

        % extract alpha signal
        alp = filtfilt(d1,data);

        % get phase of both signal
        hg_alpha_ph = angle(hilbert(hg_alpha));
        alp_ph = angle(hilbert(alp));

        % cut it to state 3
        alp_ph = alp_ph(l2+1:end,:);
        hg_alpha_ph = hg_alpha_ph(l2+1:end,:);

        % get phase difference and circular mean
        circ_diff = alp_ph - hg_alpha_ph;
        m = circ_mean(circ_diff);

        % store
        pac = cat(1,pac,m);
        alp_ph = alp_ph(:,logical(good_ch));
        hg_alpha_ph = hg_alpha_ph(:,logical(good_ch));
        alpha_phase = cat(1,alpha_phase,alp_ph);        
        hg_alpha_phase = cat(1,hg_alpha_phase,hg_alpha_ph);
    end
end

%get rid of bad channels
pac=pac(:,logical(good_ch));
pac = exp(1i*pac);
%pac_r = abs(mean(pac_r));


