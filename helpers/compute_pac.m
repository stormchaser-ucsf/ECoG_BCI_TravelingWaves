function [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2,varargin)


%load the data and extract PAC at each channel
idx=0;
pac=[];
hg_alpha_phase={};
alpha_phase={};
if length(varargin)==0
    b1=false;
else
    b1=true;
end

if b1==false
    bad_ch=[108 113 118];
    good_ch=ones(256,1);
    good_ch(bad_ch)=0;
else
    good_ch=(ones(128,1));
end

for ii=1:length(files)
    disp(ii/length(files)*100)
    loaded=1;
    try
        load(files{ii});
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
        hg = filtfilt(d2,data);hg0=hg;
        hg = abs(hilbert(hg));

        % extract the alpha component of the hg envelope
        hg_alpha = filtfilt(d1,hg);

        % extract alpha signal
        alp = filtfilt(d1,data);

        % get phase of both signal
        hg_alpha_ph = angle(hilbert(hg_alpha));
        alp_ph = angle(hilbert(alp));
        % 
        % % plotting
        % figure;
        % plot(data(1:end,200));
        % figure;hold on
        % plot(hg0(1:end,200))
        % plot(hg(1:end,200))
        % xlim([1700 2100])
        % figure;
        % plot(hg_alpha(1:end,200))
        % figure;
        % plot(alp(1:end,200))
        % xlim([1700 2100])
        % figure;
        % plot(hg_alpha_ph(:,200));
        % xlim([1700 2100])
        % figure;
        % plot(alp_ph(:,200));
        % xlim([1700 2100])


        % cut it to state 3
        alp=alp(l2+1:end,:);
        hg_alpha=hg_alpha(l2+1:end,:);
        alp_ph = alp_ph(l2+1:end,:);
        hg_alpha_ph = hg_alpha_ph(l2+1:end,:);

        % % temp plotting
        % angles = circ_diff(:,50);
        % [t,r]=rose(angles,20);
        % %subplot(4,8,i);
        % figure
        % polarplot(t,r,'LineWidth',1,'Color','k');
        % pax=gca;
        % %pax.RLim = [0 20];
        % thetaticks(0:45:315);
        % pax.ThetaAxisUnits='radians';
        % pax.FontSize=16;
        % set(gcf,'Color','w')
        % pax.RTick = [5 10 15 20 ];
        % pax.GridAlpha = 0.25;
        % pax.MinorGridAlpha = 0.25;
        % pax.ThetaMinorGrid = 'off';
        % %pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
        % pax.ThetaTickLabel = '';
        % %pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
        % pax.RTickLabel = {' ',' '};
        % pax.RAxisLocation=1;
        % pax.RAxis.LineWidth=1;
        % pax.ThetaAxis.LineWidth=1;
        % pax.LineWidth=1;
        % temp = exp(1i*angles);
        % r1 = abs(mean(temp))*max(2*r);
        % phi = angle(mean(temp));
        % hold on;
        % polarplot([phi-0.01 phi],[0 r1],'LineWidth',1.5,'Color','r');
        % %set(gcf,'PaperPositionMode','auto')
        % %set(gcf,'Position',[680.0,865,120.0,113.0])
        % title([num2str(ii) '    ' num2str(abs(mean(exp(1i*circ_diff(:,50)))))])

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

%
% % temp plotting
% tmp = (pac(:,50));
% [t,r]=rose(angle(tmp),20);
% %subplot(4,8,i);
% figure
% polarplot(t,r,'LineWidth',1,'Color','k');
% pax=gca;
% %pax.RLim = [0 20];
% thetaticks(0:45:315);
% pax.ThetaAxisUnits='radians';
% pax.FontSize=16;
% set(gcf,'Color','w')
% pax.RTick = [5 10 15 20 ];
% pax.GridAlpha = 0.25;
% pax.MinorGridAlpha = 0.25;
% pax.ThetaMinorGrid = 'off';
% %pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
% pax.ThetaTickLabel = '';
% %pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
% pax.RTickLabel = {' ',' '};
% pax.RAxisLocation=1;
% pax.RAxis.LineWidth=1;
% pax.ThetaAxis.LineWidth=1;
% pax.LineWidth=1;
% r1 = abs(mean(tmp))*max(1*r);
% phi = angle(mean(tmp));
% hold on;
% polarplot([phi-0.01 phi],[0 r1],'LineWidth',1.5,'Color','r');
% %set(gcf,'PaperPositionMode','auto')
% %set(gcf,'Position',[680.0,865,120.0,113.0])
% title(['PLV ' num2str(abs(mean((pac(:,50)))))])
