function zoom_into_ecog_grid(elecmatrix,zoom_factor)

if nargin<2
    zoom_factor=3;
end

ax = gca;
% Center of electrode grid
grid_center = mean(elecmatrix, 1);
% Current camera position and target
old_target = camtarget(ax);
old_pos = campos(ax);
% Translate camera to electrode grid while preserving view direction
shift = grid_center - old_target;
campos(ax, old_pos + shift);
camtarget(ax, grid_center);
% Zoom in
camproj(ax, 'orthographic');
camzoom(ax, zoom_factor);
axis(ax, 'vis3d');
axis(ax, 'off');
