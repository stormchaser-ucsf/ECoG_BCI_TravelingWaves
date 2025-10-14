function vortices = detect_nonoverlapping_vortices(curlMap, thresh, localFrac, minSize)
% DETECT_NONOVERLAPPING_VORTICES
% Detect strong curl peaks and assign non-overlapping symmetric masks.
%
% Inputs:
%   curlMap    : 2D array of curl values
%   strongFrac : fraction of max(|curl|) for strong peaks (e.g. 0.75)
%   localFrac  : local fraction of peak curl for extent (e.g. 0.3)
%   minSize    : minimum number of pixels per component (e.g. 5)
%
% Output:
%   vortices : struct array with fields:
%              .center [y x]
%              .Cpeak
%              .componentMask
%              .symMask
%              .bbox [xmin xmax ymin ymax]

if nargin < 2, strongFrac = 0.75; end
if nargin < 3, localFrac = 0.3; end
if nargin < 4, minSize = 5; end

[rows, cols] = size(curlMap);
vortices = struct('center', {}, 'Cpeak', {}, 'componentMask', {}, ...
                  'symMask', {}, 'bbox', {});

%% Step 1: Find strong curl peaks

strongMask = abs(curlMap) >= thresh;

localMaxMask = imregionalmax(abs(curlMap));
peakMask = strongMask & localMaxMask;
[yPeaks, xPeaks] = find(peakMask);

fprintf('Found %d strong curl peaks above %.2f.\n', numel(xPeaks), thresh);

%% Step 2: Sort peaks by descending curl magnitude
Cpeaks = abs(curlMap(sub2ind(size(curlMap), yPeaks, xPeaks)));
[~, order] = sort(Cpeaks, 'descend');
xPeaks = xPeaks(order);
yPeaks = yPeaks(order);
Cpeaks = Cpeaks(order);

%% Step 3: Initialize global mask to track assigned pixels
assignedMask = false(size(curlMap));
for k = 1:numel(xPeaks)
    x0 = xPeaks(k);
    y0 = yPeaks(k);
    Cpeak = Cpeaks(k);
    thresh = localFrac * Cpeak;
    %thresh = localFrac ;


    % Local threshold
    maskThresh = abs(curlMap) >= thresh;

    % Connected component containing this peak
    CC = bwconncomp(maskThresh, 8);
    centerIdx = sub2ind(size(curlMap), y0, x0);
    componentMask = false(size(curlMap));
    for i = 1:CC.NumObjects
        if ismember(centerIdx, CC.PixelIdxList{i})
            componentMask(CC.PixelIdxList{i}) = true;
            break;
        end
    end

    % Skip if component too small
    if sum(componentMask(:)) < minSize
        continue;
    end

    % Compute symmetric rectangular mask
    [yy, xx] = find(componentMask);
    dx = xx - x0; dy = yy - y0;
    x_half = max(abs([min(dx) max(dx)]));
    y_half = max(abs([min(dy) max(dy)]));
    [X, Y] = meshgrid(1:cols, 1:rows);
    symMask = abs(X - x0) <= x_half & abs(Y - y0) <= y_half;

    %  Remove any rows/columns containing other vortex centers
    otherCenters = [yPeaks, xPeaks];
    for j = 1:numel(xPeaks)
        if j == k, continue; end
        yc = otherCenters(j,1);
        xc = otherCenters(j,2);
        if symMask(yc, xc)
            % Remove entire row and column of that center
            symMask(yc, :) = false;
            symMask(:, xc) = false;
        end
    end

    % Enforce non-overlap with stronger vortices
    symMask = symMask & ~assignedMask;

    if sum(symMask(:)) < minSize
        continue;
    end

    % Update global assignment
    assignedMask = assignedMask | symMask;

    bbox = [max(1, x0 - x_half), min(cols, x0 + x_half), ...
            max(1, y0 - y_half), min(rows, y0 + y_half)];

    vortices(end+1).center = [y0, x0];
    vortices(end).Cpeak = Cpeak;
    vortices(end).componentMask = componentMask;
    vortices(end).symMask = symMask;
    vortices(end).bbox = bbox;
end


end



