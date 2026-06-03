


dirn=pwd;

if ispc
    addpath('C:\Users\nikic\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016')
    cd('C:\Users\nikic\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016')

else
    addpath('/home/user/Documents/MATLAB/ctmr_gauss_plot_April2016/ctmr_gauss_plot_April2016/')    
    cd('/home/user/Documents/MATLAB/ctmr_gauss_plot_April2016/ctmr_gauss_plot_April2016/bravo6')
end

load(fullfile(pwd,'Meshes','B6_lh_pial.mat'))
%load('BRAVO1_elecs_all')
load('bravo6_grid_rt_reordered.mat')


imaging_path='C:\Users\Nikhlesh\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016';
%load('loc_colormap_thresh')

ch=1:size(anatomy,1);
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
% To plot electrodes with numbers, use the following, as example:
e_h = el_add(elecmatrix([1:length(ch)],:), 'color', 'b', 'numbers', ch);
% Or you can just plot them without labels with the default color
%e_h = el_add(elecmatrix([1:length(ch)],:), 'color', 'w', 'msize',2);
%e_h = el_add(elecmatrix(1:64,:)); % only loading 48 electrode data
set(gcf,'Color','w')
cd(dirn)



% plotting by ROI
if ispc
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
else
    addpath('/home/user/Documents/MATLAB/DrosteEffect-BrewerMap-5b84f95')
end
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
set(gcf,'Color','w')
rois = unique(anatomy(:,4));
colmap = parula(length(rois));
%colmap = brewermap(length(rois),'Set1');
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

% make changes to ROI list
anatomy(49,4) = anatomy(218,4);
anatomy(51,4) = anatomy(218,4);
anatomy(47,4) = anatomy(218,4);
anatomy(43,4) = anatomy(218,4);
anatomy(39,4) = anatomy(218,4);
anatomy(36,4) = anatomy(218,4);
anatomy(252,4) = anatomy(218,4);
anatomy(176,4) = anatomy(218,4);
anatomy(40,4) = anatomy(218,4);

anatomy(112,4) = anatomy(119,4);
anatomy(108,4) = anatomy(119,4);
anatomy(76,4) = anatomy(119,4);
anatomy(112,4) = anatomy(119,4);
anatomy(112,4) = anatomy(119,4);

anatomy(72,4) = anatomy(107,4);

anatomy(250,4) = anatomy(224,4);
anatomy(76,4) = anatomy(224,4);

% plot again
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
set(gcf,'Color','w')
rois = unique(anatomy(:,4));
colmap = parula(length(rois));
%colmap = brewermap(length(rois),'Set1');
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


% plotting with a color bar denoting the phase values and the radius
% denoting the amplitude values.
ph=[linspace(pi,-pi,128)];
phMap = linspace(-pi,pi,128)';
ChColorMap=parula(128);
val = rand(128,1);
ch=1:128;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
for j=1:length(val)
    ms = val(j)*(10-1)+1;
    [aa bb]=min(abs(ph(j) - phMap));
    c=ChColorMap(bb,:);
    e_h = el_add(elecmatrix(j,:), 'color', c,'msize',ms);
end

