function pax = plot_phases(data)
%function pax = plot_phases(data)
% INPUT: angular data


% temp plotting
tmp = data;

[t,r]=rose((tmp),20);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:45:315);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = '';
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
temp = exp(1i*tmp);
r1 = abs(mean(temp))*max(1*r);
phi = angle(mean(temp));
hold on;
polarplot([phi-0.01 phi],[0 r1],'LineWidth',1.5,'Color','r');
title(['Preferred angle  ' num2str(phi*180/pi)])
%set(gcf,'PaperPositionMode','auto')
%set(gcf,'Position',[680.0,865,120.0,113.0])
%title(['PLV ' num2str(abs(mean((pac(:,50)))))])
