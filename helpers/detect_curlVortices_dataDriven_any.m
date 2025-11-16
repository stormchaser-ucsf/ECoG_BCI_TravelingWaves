function vortices = detect_curlVortices_dataDriven_any(curlMap, xph,xphs,Thresh)
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

if nargin < 4, Thresh = 0.75; end


[rows, cols] = size(curlMap);
vortices = struct('center', {}, 'Cpeak', {}, 'componentMask', {}, ...
                  'symMask', {}, 'bbox', {});

%% Step 1: Find strong curl peaks

strongMask = (curlMap) >= Thresh;
localMaxMask = imregionalmax((curlMap));
peakMask = strongMask & localMaxMask;
[yPeaks, xPeaks] = find(peakMask);

% if any of the peaks are at the edges, discard
% if any(yPeaks == 1) || any(yPeaks == rows) || any(xPeaks == 1) ...
%         || any(xPeaks == cols) || numel(xPeaks)==0
%     fprintf('No curl peaks above threshold or in grid \n');
% else
%     fprintf('Found %d strong curl peaks above %.2f.\n', numel(xPeaks), Thresh);
% end


%% (NEW)

% figure;
% imagesc(curlMap)
% colorbar 
% hold on
vortices={};
corr_total=[];
for k=1:numel(xPeaks)
    x0 = xPeaks(k);
    y0 = yPeaks(k);

    % initial 3X3
    cmin = x0-1;
    cmax = x0+1;
    rmin = y0-1;
    rmax = y0+1;

    if cmin==0 || cmax > cols || rmin ==0 || rmax>rows
        continue;
    end

    % perform circular linear correlation w/ init 3X3
    pl = angle(double(xphs(rmin:rmax, cmin:cmax))); %xphs for init
    [cc1,pv,center_point] = phase_correlation_rotation( pl,...
        double(curlMap(rmin:rmax, cmin:cmax)),[],-1);
    corr_val_init = abs(cc1);
    pval_init = pv;
    loop_stay = true;
    corr_val=corr_val_init;
    pval=1;
   
    corr_val_update=[corr_val_init];
    pval_update=[pval_init];
    bbox_update={};kk=1;
    bbox_update(kk).r = [rmin rmax];
    bbox_update(kk).c = [cmin cmax];
    iterr=0;
    while loop_stay
        iterr=iterr+1;

        % expand bounding box  
        bbox={};
        [bbox] = expand_bound_box_all(bbox,cmin,cmax,rmin,rmax,...
            rows,cols);

        % compute correlation for bbox
        %lt
        collt_corr=0;pv_collt=1;
        if bbox.collt.flag
            c = bbox.collt.cols;
            r = bbox.collt.rows;
            pl = angle(double(xph(r(1):r(2),c(1):c(2))));
            [cc1,pv_collt,center_point] = phase_correlation_rotation( pl,...
                double(curlMap(r(1):r(2),c(1):c(2))),[],-1);
            collt_corr = (cc1);
        end
        
        %rt
        colrt_corr=0;pv_colrt=1;
        if bbox.colrt.flag
            c = bbox.colrt.cols;
            r = bbox.colrt.rows;
            pl = angle(double(xph(r(1):r(2),c(1):c(2))));
            [cc1,pv_colrt,center_point] = phase_correlation_rotation( pl,...
                double(curlMap(r(1):r(2),c(1):c(2))),[],-1);
            colrt_corr = (cc1);
        end

        %up
        rowu_corr=0;pv_rowu=1;
        if bbox.rowu.flag
            c = bbox.rowu.cols;
            r = bbox.rowu.rows;
            pl = angle(double(xph(r(1):r(2),c(1):c(2))));
            [cc1,pv_rowu,center_point] = phase_correlation_rotation( pl,...
                double(curlMap(r(1):r(2),c(1):c(2))),[],-1);
            rowu_corr = (cc1);
        end

        %down
        rowd_corr=0;pv_rowd=1;
        if bbox.rowd.flag
            c = bbox.rowd.cols;
            r = bbox.rowd.rows;
            pl = angle(double(xph(r(1):r(2),c(1):c(2))));
            [cc1,pv_rowd,center_point] = phase_correlation_rotation( pl,...
                double(curlMap(r(1):r(2),c(1):c(2))),[],-1);
            rowd_corr = (cc1);
        end
        
      
        total_corr = abs([collt_corr colrt_corr rowu_corr rowd_corr]);
        %disp(total_corr)
        total_pval = [pv_collt pv_colrt pv_rowu pv_rowd];
        [aa bb]=max(total_corr);
        corr_val_update = [corr_val_update total_corr(bb)];
        pval_update = [pval_update total_pval(bb)];
        if total_pval(bb) <= 0.05
            corr_val = total_corr(bb);
            pval = total_pval(bb);
            switch bb
                case 1 % lt
                    
                    cmin = bbox.collt.cols(1);
                    cmax = bbox.collt.cols(2);
                    rmin = bbox.collt.rows(1);
                    rmax = bbox.collt.rows(2);

                case 2 % rt
                   
                    cmin = bbox.colrt.cols(1);
                    cmax = bbox.colrt.cols(2);
                    rmin = bbox.colrt.rows(1);
                    rmax = bbox.colrt.rows(2);

                case 3 % up
                   
                    cmin = bbox.rowu.cols(1);
                    cmax = bbox.rowu.cols(2);
                    rmin = bbox.rowu.rows(1);
                    rmax = bbox.rowu.rows(2);

                case 4 % down
                    
                    cmin = bbox.rowd.cols(1);
                    cmax = bbox.rowd.cols(2);
                    rmin = bbox.rowd.rows(1);
                    rmax = bbox.rowd.rows(2);
            end
            kk=kk+1;
            bbox_update(kk).r = [rmin rmax];
            bbox_update(kk).c = [cmin cmax];            
        else
            kk=kk+1;
            bbox_update(kk).r = [rmin rmax];
            bbox_update(kk).c = [cmin cmax];
            loop_stay = false;
        end
        %disp([cmin cmax rmin rmax])
            
    end
    [aa bb]=max(corr_val_update);
    cmin = bbox_update(bb).c(1);cmax = bbox_update(bb).c(2);
    rmin = bbox_update(bb).r(1);rmax = bbox_update(bb).r(2);
    % rectangle('Position', [cmin, rmin, cmax - cmin, rmax - rmin], ...
    %       'EdgeColor', 'r', 'LineWidth', 2);  
    vortices(k).corr = corr_val_update(bb);
    vortices(k).pval = pval_update(bb);
    vortices(k).cols = [cmin cmax];
    vortices(k).rows = [rmin rmax];
    corr_total(k) = corr_val;
end
%title(['Average correlation of ' num2str(mean( [vortices(1:end).corr]))]) 

%% (OLD) STEP 2: For each curl peak, use circular linear correlation to expand 
% in any direction around the center peak and find extent


% 
% figure;
% imagesc(curlMap)
% colorbar 
% hold on
% vortices={};
% corr_total=[];
% for k=1:numel(xPeaks)
%     x0 = xPeaks(k);
%     y0 = yPeaks(k);
% 
%     % initial 3X3
%     cmin = x0-1;
%     cmax = x0+1;
%     rmin = y0-1;
%     rmax = y0+1;
% 
%     if cmin==0 || cmax > cols || rmin ==0 || rmax>rows
%         continue;
%     end
% 
%     % perform circular linear correlation w/ init 3X3
%     pl = angle(double(xph(rmin:rmax, cmin:cmax)));
%     [cc1,pv,center_point] = phase_correlation_rotation( pl,...
%         double(curlMap(rmin:rmax, cmin:cmax)),[],-1);
%     corr_val_init = abs(cc1);
%     pval_init = pv;
%     loop_stay = true;
%     corr_val=corr_val_init;
%     pval=1;
% 
% 
% 
%     corr_val_update=[corr_val_init];
%     pval_update=[pval_init];
%     bbox_update={};kk=1;
%     bbox_update(kk).r = [rmin rmax];
%     bbox_update(kk).c = [cmin cmax];
%     while loop_stay
% 
%         % expand bounding box  
%         bbox={};
%         [bbox] = expand_bound_box_all(bbox,cmin,cmax,rmin,rmax,...
%             rows,cols);
% 
%         % compute correlation for bbox
%         %lt
%         collt_corr=0;pv_collt=1;
%         if bbox.collt.flag
%             c = bbox.collt.cols;
%             r = bbox.collt.rows;
%             pl = angle(double(xph(r(1):r(2),c(1):c(2))));
%             [cc1,pv_collt,center_point] = phase_correlation_rotation( pl,...
%                 double(curlMap(r(1):r(2),c(1):c(2))),[],-1);
%             collt_corr = (cc1);
%         end
% 
%         %rt
%         colrt_corr=0;pv_colrt=1;
%         if bbox.colrt.flag
%             c = bbox.colrt.cols;
%             r = bbox.colrt.rows;
%             pl = angle(double(xph(r(1):r(2),c(1):c(2))));
%             [cc1,pv_colrt,center_point] = phase_correlation_rotation( pl,...
%                 double(curlMap(r(1):r(2),c(1):c(2))),[],-1);
%             colrt_corr = (cc1);
%         end
% 
%         %up
%         rowu_corr=0;pv_rowu=1;
%         if bbox.rowu.flag
%             c = bbox.rowu.cols;
%             r = bbox.rowu.rows;
%             pl = angle(double(xph(r(1):r(2),c(1):c(2))));
%             [cc1,pv_rowu,center_point] = phase_correlation_rotation( pl,...
%                 double(curlMap(r(1):r(2),c(1):c(2))),[],-1);
%             rowu_corr = (cc1);
%         end
% 
%         %down
%         rowd_corr=0;pv_rowd=1;
%         if bbox.rowd.flag
%             c = bbox.rowd.cols;
%             r = bbox.rowd.rows;
%             pl = angle(double(xph(r(1):r(2),c(1):c(2))));
%             [cc1,pv_rowd,center_point] = phase_correlation_rotation( pl,...
%                 double(curlMap(r(1):r(2),c(1):c(2))),[],-1);
%             rowd_corr = (cc1);
%         end
% 
% 
%         total_corr = abs([collt_corr colrt_corr rowu_corr rowd_corr]);
%         %disp(total_corr)
%         total_pval = [pv_collt pv_colrt pv_rowu pv_rowd];
%         [aa bb]=max(total_corr);
%         corr_val_update = [corr_val_update total_corr(bb)];
%         pval_update = [pval_update total_pval(bb)];
%         if total_pval(bb) <= 0.05
%             corr_val = total_corr(bb);
%             pval = total_pval(bb);
%             switch bb
%                 case 1 % lt
% 
%                     cmin = bbox.collt.cols(1);
%                     cmax = bbox.collt.cols(2);
%                     rmin = bbox.collt.rows(1);
%                     rmax = bbox.collt.rows(2);
% 
%                 case 2 % rt
% 
%                     cmin = bbox.colrt.cols(1);
%                     cmax = bbox.colrt.cols(2);
%                     rmin = bbox.colrt.rows(1);
%                     rmax = bbox.colrt.rows(2);
% 
%                 case 3 % up
% 
%                     cmin = bbox.rowu.cols(1);
%                     cmax = bbox.rowu.cols(2);
%                     rmin = bbox.rowu.rows(1);
%                     rmax = bbox.rowu.rows(2);
% 
%                 case 4 % down
% 
%                     cmin = bbox.rowd.cols(1);
%                     cmax = bbox.rowd.cols(2);
%                     rmin = bbox.rowd.rows(1);
%                     rmax = bbox.rowd.rows(2);
%             end
%             kk=kk+1;
%             bbox_update(kk).r = [rmin rmax];
%             bbox_update(kk).c = [cmin cmax];            
%         else
%             kk=kk+1;
%             bbox_update(kk).r = [rmin rmax];
%             bbox_update(kk).c = [cmin cmax];
%             loop_stay = false;
%         end
% 
%     end
%     [aa bb]=max(corr_val_update);
%     cmin = bbox_update(bb).c(1);cmax = bbox_update(bb).c(2);
%     rmin = bbox_update(bb).r(1);rmax = bbox_update(bb).r(2);
%     rectangle('Position', [cmin, rmin, cmax - cmin, rmax - rmin], ...
%           'EdgeColor', 'r', 'LineWidth', 2);  
%     vortices(k).corr = corr_val_update(bb);
%     vortices(k).pval = pval_update(bb);
%     vortices(k).cols = [cmin cmax];
%     vortices(k).rows = [rmin rmax];
%     corr_total(k) = corr_val;
% end
% title(['Average correlation of ' num2str(mean( [vortices(1:end).corr]))]) 
% 
% 
% %% Step 2: Sort peaks by descending curl magnitude
% %Cpeaks = abs(curlMap(sub2ind(size(curlMap), yPeaks, xPeaks)));
% Cpeaks = (curlMap(sub2ind(size(curlMap), yPeaks, xPeaks)));
% [~, order] = sort(Cpeaks, 'descend');
% xPeaks = xPeaks(order);
% yPeaks = yPeaks(order);
% Cpeaks = Cpeaks(order);
% 
% %% Step 3: Initialize global mask to track assigned pixels
% assignedMask = false(size(curlMap));
% for k = 1:numel(xPeaks)
%     x0 = xPeaks(k);
%     y0 = yPeaks(k);
%     Cpeak = Cpeaks(k);
%     thresh = localFrac * Cpeak;
%     %thresh = localFrac ;
% 
% 
%     % Local threshold
%     if Cpeak>0
%     maskThresh = (curlMap) >= thresh;
%     else
%         maskThresh = (curlMap) <= thresh;
%     end
% 
%     % Connected component containing this peak
%     CC = bwconncomp(maskThresh, 8);
%     centerIdx = sub2ind(size(curlMap), y0, x0);
%     componentMask = false(size(curlMap));
%     for i = 1:CC.NumObjects
%         if ismember(centerIdx, CC.PixelIdxList{i})
%             componentMask(CC.PixelIdxList{i}) = true;
%             break;
%         end
%     end
% 
%     % Skip if component too small
%     if sum(componentMask(:)) < minSize
%         continue;
%     end
% 
%     % Compute symmetric rectangular mask
%     [yy, xx] = find(componentMask);
%     dx = xx - x0; dy = yy - y0;
%     x_half = max(abs([min(dx) max(dx)]));
%     y_half = max(abs([min(dy) max(dy)]));
%     [X, Y] = meshgrid(1:cols, 1:rows);
%     symMask = abs(X - x0) <= x_half & abs(Y - y0) <= y_half;
% 
%     % %  Remove any rows/columns containing other vortex centers
%     % otherCenters = [yPeaks, xPeaks];
%     % for j = 1:numel(xPeaks)
%     %     if j == k, continue; end
%     %     yc = otherCenters(j,1);
%     %     xc = otherCenters(j,2);
%     %     if symMask(yc, xc)
%     %         % Remove entire row and column of that center
%     %         symMask(yc, :) = false;
%     %         symMask(:, xc) = false;
%     %     end
%     % end
% 
%     % Enforce non-overlap with stronger vortices
%     symMask = symMask & ~assignedMask;
% 
%     if sum(symMask(:)) < minSize
%         continue;
%     end
% 
%     % Update global assignment
%     assignedMask = assignedMask | symMask;
% 
%     bbox = [max(1, x0 - x_half), min(cols, x0 + x_half), ...
%             max(1, y0 - y_half), min(rows, y0 + y_half)];
% 
%     vortices(end+1).center = [y0, x0];
%     vortices(end).Cpeak = Cpeak;
%     vortices(end).componentMask = componentMask;
%     vortices(end).symMask = symMask;
%     vortices(end).bbox = bbox;
% end


end



