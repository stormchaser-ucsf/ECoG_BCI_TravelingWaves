function [center, componentMask, symMask, bbox] = rotational_extent_from_curl(curlMap, center)
% ROTATIONAL_EXTENT_FROM_CURL
% Computes the rotational extent around a center using curl.
%
% Inputs:
%   curlMap : 2D matrix of curl magnitudes (already computed)
%   center  : optional [y0, x0] coordinates. If empty or NaN, the function
%             will pick the location of maximum curl.
%
% Outputs:
%   center        : [y0, x0] used as rotation center
%   componentMask : raw connected component at threshold (30% of peak)
%   symMask       : symmetric rectangular mask around center
%   bbox          : [xmin xmax ymin ymax] of symmetric mask

%% Parameters
lowFrac = 0.3;  % fraction of peak curl to define threshold
minArea = 4;    % optional: minimum pixels to keep component

[rows, cols] = size(curlMap);

%% Step 1: Find center if not provided
if nargin < 2 || isempty(center) || any(isnan(center))
    [~, idx] = max(abs(curlMap(:)));
    [y0, x0] = ind2sub(size(curlMap), idx);
else
    x0 = round(center(2));
    y0 = round(center(1));
end
center = [y0, x0];

%% Step 2: Threshold curl
thresh = lowFrac * abs(curlMap(y0, x0));
maskThresh = abs(curlMap) >= thresh;

%% Step 3: Keep connected component containing center
CC = bwconncomp(maskThresh, 8);  % 8-connectivity
componentMask = false(size(curlMap));
centerIdx = sub2ind(size(curlMap), y0, x0);

for k = 1:CC.NumObjects
    if ismember(centerIdx, CC.PixelIdxList{k})
        componentMask(CC.PixelIdxList{k}) = true;
        break;
    end
end

% Optional: remove tiny regions
componentMask = bwareaopen(componentMask, minArea);

%% Step 4: Symmetric rectangular mask

if sum(componentMask(:))>0
    [yy, xx] = find(componentMask);

    dx = xx - x0;
    dy = yy - y0;

    x_half = max(abs([min(dx) max(dx)]));
    y_half = max(abs([min(dy) max(dy)]));

    xmin = max(1, x0 - x_half); xmax = min(cols, x0 + x_half);
    ymin = max(1, y0 - y_half); ymax = min(rows, y0 + y_half);

    [X, Y] = meshgrid(1:cols, 1:rows);
    symMask = abs(X - x0) <= x_half & abs(Y - y0) <= y_half;

    bbox = [xmin xmax ymin ymax];

else
    symMask = zeros(rows,cols);
    bbox = [];
end
