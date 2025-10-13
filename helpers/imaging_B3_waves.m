

% 
% dirn=pwd;
% addpath(genpath('C:\Users\nikic\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016'))
% cd('C:\Users\nikic\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016\Bravo3')

load('BRAVO3_lh_pial')
%load('BRAVO2_elecs_all2')
load('grid.mat')


ch=1:size(anatomy,1);
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
% To plot electrodes with numbers, use the following, as example:
e_h = el_add(elecmatrix([1:length(ch)],:), 'color', 'b', 'numbers', ch);
% Or you can just plot them without labels with the default color
%e_h = el_add(elecmatrix([1:length(ch)],:), 'color', 'r', 'msize',12);
%e_h = el_add(elecmatrix(1:64,:)); % only loading 48 electrode data
set(gcf,'Color','w')
%cd(dirn)


% plotting by ROI
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
set(gcf,'Color','w')
rois = unique(anatomy(:,4));
colmap = parula(length(rois));
for j=1:size(anatomy,1)
    ch_roi = anatomy{j,4};
    for i=1:length(rois)
        if strcmp(ch_roi,rois{i})
            col = colmap(i,:);
            e_h = el_add(elecmatrix(j,:), ...
                'color', col, 'msize',8);
        end
    end 
end


% plotting by ROI after correction
load('anatomy_B3')
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
set(gcf,'Color','w')
rois = unique(anatomy_B3(:,4));
colmap = parula(length(rois));
for j=1:size(anatomy_B3,1)
    ch_roi = anatomy_B3{j,4};
    for i=1:length(rois)
        if strcmp(ch_roi,rois{i})
            col = colmap(i,:);
            e_h = el_add(elecmatrix(j,:), ...
                'color', col, 'msize',8);
        end
    end 
end



% plotting with a color bar denoting the phase values and the radius
% denoting the amplitude values.
ph=[linspace(pi,-pi,253)];
phMap = linspace(-pi,pi,253)';
ChColorMap=parula(253);
val = rand(253,1);
ch=1:253;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
for j=1:length(val)
    ms = val(j)*(10-1)+1;
    [aa bb]=min(abs(ph(j) - phMap));
    c=ChColorMap(bb,:);
    e_h = el_add(elecmatrix(j,:), 'color', c,'msize',ms);
end


grid_layout=[];
for i=1:23:253
    grid_layout = [grid_layout (i:i+22)'];
end
grid_layout = fliplr(grid_layout');

