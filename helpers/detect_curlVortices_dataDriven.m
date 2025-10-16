function vortices = detect_curlVortices_dataDriven(curlMap, Thresh, minSize)
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

if nargin < 2, Thresh = 0.75; end
if nargin < 3, minSize = 9; end

[rows, cols] = size(curlMap);
vortices = struct('center', {}, 'Cpeak', {}, 'componentMask', {}, ...
                  'symMask', {}, 'bbox', {});

%% Step 1: Find strong curl peaks

strongMask = abs(curlMap) >= Thresh;

localMaxMask = imregionalmax(abs(curlMap));
peakMask = strongMask & localMaxMask;
[yPeaks, xPeaks] = find(peakMask);

% if any of the peaks are at the edges, discard
if any(yPeaks == 1) || any(yPeaks == rows) || any(xPeaks == 1) ...
        || any(xPeaks == cols) || numel(xPeaks)==0
    fprintf('No curl peaks above threshold or in grid \n');
else
    fprintf('Found %d strong curl peaks above %.2f.\n', numel(xPeaks), Thresh);
end


%% STEP 2: For each curl peak, use circular linear correlation to expand 
% in any direction around the center peak and find extent


for k=1:numel(xPeaks)
    x0 = xPeaks(k);
    y0 = yPeaks(k);

    % initial 3X3
    cmin = x0-1;
    cmax = x0+1;
    rmin = y0-1;
    rmax = y0+1;

    % perform circular linear correlation w/ init 3X3
    pl = angle(double(xphs(rmin:rmax, cmin:cmax)));
    [cc1,pv,center_point] = phase_correlation_rotation( pl,...
        double(curl_val0(rmin:rmax, cmin:cmax)),[],-1);
    corr_val_init = abs(cc1);
    pval_init = pv;
    loop_stay = true;

    row_flag=false;col_flag=false;symm_flag=false;
    bbox={};
    corr_val_update=[];
    pval_update=[];
    while loop_stay

        % expand bounding box       
        [bbox] = expand_bound_box(bbox,cmin,cmax,rmin,rmax,...
            row_flag,col_flag,symm_flag,rows,cols);

        % compute correlation for bbox
        symm_corr=0;pv_symm=1;
        row_corr=0;pv_row=1;
        col_corr=0;pv_col=1;
        if bbox.symm.flag
            c = bbox.symm.cols;
            r = bbox.symm.rows;
            pl = angle(double(xphs(r(1):r(2),c(1):c(2))));
            [cc1,pv_symm,center_point] = phase_correlation_rotation( pl,...
                double(curl_val0(r(1):r(2),c(1):c(2))),[],-1);
            symm_corr = abs(cc1);
        end
        if bbox.row.flag
            c = bbox.row.cols;
            r = bbox.row.rows;
            pl = angle(double(xphs(r(1):r(2),c(1):c(2))));
            [cc1,pv_row,center_point] = phase_correlation_rotation( pl,...
                double(curl_val0(r(1):r(2),c(1):c(2))),[],-1);
            row_corr = abs(cc1);
        end
        if bbox.col.flag
            c = bbox.col.cols;
            r = bbox.col.rows;
            pl = angle(double(xphs(r(1):r(2),c(1):c(2))));
            [cc1,pv_col,center_point] = phase_correlation_rotation( pl,...
                double(curl_val0(r(1):r(2),c(1):c(2))),[],-1);
            col_corr = abs(cc1);
        end
        total_corr = [symm_corr row_corr col_corr];
        total_pval = [pv_symm pv_row pv_col];
        [aa bb]=max(total_corr);
        corr_val_update = [corr_val_update total_corr(bb)];
        pval_update = [pval_update total_pval(bb)];
        if total_pval(bb)> 0.05
            loop_stay = false;
        else
            switch bb
                case 1 % symm
                    symm_flag=true;
                    cmin = bbox.symm.cols(1);
                    cmax = bbox.symm.cols(2);
                    rmin = bbox.symm.rows(1);
                    rmax = bbox.symm.rows(2);

                case 2 % row
                    row_flag=true;
                    cmin = bbox.row.cols(1);
                    cmax = bbox.row.cols(2);
                    rmin = bbox.row.rows(1);
                    rmax = bbox.row.rows(2);

                case 3 % col
                    col_flag=true;
                    cmin = bbox.col.cols(1);
                    cmax = bbox.col.cols(2);
                    rmin = bbox.col.rows(1);
                    rmax = bbox.col.rows(2);
            end
        end
    end
    
end

%% Step 2: Sort peaks by descending curl magnitude
%Cpeaks = abs(curlMap(sub2ind(size(curlMap), yPeaks, xPeaks)));
Cpeaks = (curlMap(sub2ind(size(curlMap), yPeaks, xPeaks)));
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
    if Cpeak>0
    maskThresh = (curlMap) >= thresh;
    else
        maskThresh = (curlMap) <= thresh;
    end

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

    % %  Remove any rows/columns containing other vortex centers
    % otherCenters = [yPeaks, xPeaks];
    % for j = 1:numel(xPeaks)
    %     if j == k, continue; end
    %     yc = otherCenters(j,1);
    %     xc = otherCenters(j,2);
    %     if symMask(yc, xc)
    %         % Remove entire row and column of that center
    %         symMask(yc, :) = false;
    %         symMask(:, xc) = false;
    %     end
    % end

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



