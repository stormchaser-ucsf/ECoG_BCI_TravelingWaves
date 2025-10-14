function vortices = detect_strong_rotations(curlMap, thresh, localFrac)
% DETECT_STRONG_ROTATIONS
% Detects strong curl peaks (> strongFrac * max) and extracts their
% rotational extents (>= localFrac * local peak value).
%
% Inputs:
%   curlMap     : 2D array of curl values
%   strongFrac  : fraction of global max for initial strong-peak detection (e.g. 0.75)
%   localFrac   : fraction of local peak value for extent (e.g. 0.3)
%
% Output:
%   vortices : struct array with fields:
%              .center [y0 x0]
%              .Cpeak
%              .componentMask
%              .symMask
%              .bbox [xmin xmax ymin ymax]

if nargin < 2, strongFrac = 0.75; end
if nargin < 3, localFrac  = 0.3;  end

[rows, cols] = size(curlMap);
vortices = struct('center', {}, 'Cpeak', {}, 'componentMask', {}, 'symMask', {}, 'bbox', {});

%% Step 1: Find all strong peaks
%maxCurl = max(abs(curlMap(:)));
strongMask = abs(curlMap) >= thresh;

% Restrict to local maxima (so we don't pick large flat areas)
localMaxMask = imregionalmax(abs(curlMap));
peakMask = strongMask & localMaxMask;

[yPeaks, xPeaks] = find(peakMask);

fprintf('Found %d strong curl peaks above %.2f \n', numel(xPeaks), strongFrac);

%% Step 2: For each strong peak
for k = 1:numel(xPeaks)
    x0 = xPeaks(k);
    y0 = yPeaks(k);
    Cpeak = abs(curlMap(y0, x0));
    local_thresh = localFrac * Cpeak;

    % Step 3: threshold field around this peak
    maskThresh = abs(curlMap) >= local_thresh;

    % Step 4: connected component containing this peak
    CC = bwconncomp(maskThresh, 8);
    centerIdx = sub2ind(size(curlMap), y0, x0);
    componentMask = false(size(curlMap));
    for i = 1:CC.NumObjects
        if ismember(centerIdx, CC.PixelIdxList{i})
            componentMask(CC.PixelIdxList{i}) = true;
            break;
        end
    end

    % Step 5: Compute symmetric rectangular mask
    [yy, xx] = find(componentMask);
    dx = xx - x0; dy = yy - y0;
    x_half = max(abs([min(dx) max(dx)]));
    y_half = max(abs([min(dy) max(dy)]));

    [X, Y] = meshgrid(1:cols, 1:rows);
    symMask = abs(X - x0) <= x_half & abs(Y - y0) <= y_half;

    bbox = [max(1, x0 - x_half), min(cols, x0 + x_half), ...
            max(1, y0 - y_half), min(rows, y0 + y_half)];

    vortices(k).center = [y0, x0];
    vortices(k).Cpeak = Cpeak;
    vortices(k).componentMask = componentMask;
    vortices(k).symMask = symMask;
    vortices(k).bbox = bbox;
end
end
