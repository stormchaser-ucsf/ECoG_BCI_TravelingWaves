function [state_pow] = get_state_pow(files,bpFilt)
%function [state_pow] = get_state_pow(files,bpFilt)

alp_power={};
sizes=[];
for ii=1:length(files)
    disp(ii/length(files)*100)
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
        sizes =[sizes; [l1 l2 l3 l4]];

        data = [data1;data2;data3;data4];
        %data = [data1;data2]; % only state 1 and 2

        % extract alpha envelope
        alp = filter(bpFilt,data);
        alp_pow = abs(hilbert(alp));

        % % plotting example
        % figure;hold on
        % plot(alp(:,70),'LineWidth',1)
        % plot(alp_pow(:,70),'LineWidth',1)
        % h=vline(cumsum([l1 l2 l3 l4]),'m');
        % set(h,'LineWidth',1);
        % h=hline(0);
        % set(h,'Color','m')
        % set(h,'LineWidth',1);
        % ylabel('uV')
        % xlabel('Time (ms)')
        % plot_beautify
        % xticks(0:1000:15000)


        % z-score to the first 1000ms
        m = mean(alp_pow(1:1000,:),1);
        s = std(alp_pow(1:1000,:),1);
        alp_pow = (alp_pow-m)./s;

        % store
        alp_power = cat(1,alp_power,alp_pow);
    end
end


% power across trials in the various states
% corr_len = median(sizes);
% pow=[];ch=170;
% for i=1:length(alp_power)
%     tmp=alp_power{i};
%     s= sizes(i,:);
%     s = cumsum(s);
%     s1 = tmp(1:s(1),ch);
%     s1=mean(s1,1);
%     s2 = tmp(s(1)+1:s(2),ch);
%     s2 = mean(s2,1);
%     s3 = tmp(s(2)+1:s(3),ch);
%     s3 = mean(s3,1);
%     s4 = tmp(s(3)+1:s(4),ch);
%     s4 = mean(s4,1);
%     pow = [pow; [s1 s2 s3 s4]];
% end
% figure;boxplot(pow)
% h=hline(0,'--r');
% set(h,'LineWidth',1)
% xticklabels({'S1','S2','S3','S4'})
% ylabel('Alpha power Z score relative to S1')
% plot_beautify
% title(['Channel ' num2str(ch)])

% doing it now across channels
corr_len = median(sizes);
pow_channels=[];
for ch=1:256
    if sum(ch==[118 113 108])==0
        pow=[];
        for i=1:length(alp_power)
            tmp=alp_power{i};
            s= sizes(i,:);
            s = cumsum(s);
            s1 = tmp(1:s(1),ch);
            s1=mean(s1,1);
            s2 = tmp(s(1)+1:s(2),ch);
            s2 = mean(s2,1);
            s3 = tmp(s(2)+1:s(3),ch);
            s3 = mean(s3,1);
            s4 = tmp(s(3)+1:s(4),ch);
            s4 = mean(s4,1);
            pow = [pow; [s1 s2 s3 s4]];
        end
        pow_channels =[pow_channels;median(pow)];
    end
end
figure;boxplot(pow_channels)
h=hline(0,'--r');
set(h,'LineWidth',1)
xticklabels({'S1','S2','S3','S4'})
ylabel('Alpha power Z score relative to S1')
plot_beautify
title(['All Channels']);

state_pow = pow_channels;

