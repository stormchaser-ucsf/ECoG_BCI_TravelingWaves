function [out] = compute_wave_stats(xph,XX,YY,plot_true)

% smooth phasors to get smoothed estimates of phase
M = real(xph);
N = imag(xph);
if plot_true
    figure
    quiver( XX, YY, M, N, 0.5, 'k', 'linewidth', 2 );
    set( gca, 'fontname', 'arial', 'fontsize', 14, 'ydir', 'reverse' );
    title('Phasor eigmap')
end
M = smoothn(M,'robust'); %dx
N = smoothn(N,'robust'); %dy
if plot_true
    figure
    quiver( XX, YY, M, N, 0.5, 'k', 'linewidth', 2 );
    set( gca, 'fontname', 'arial', 'fontsize', 14, 'ydir', 'reverse' );
    title('Smoothed Phasor eigmap')
end
xphs = M + 1j*N; % smoothed phasor field: gets smoothed estimates of phase

% compute gradient
pixel_spacing=1;
sign_IF=-1;
[pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN( xphs, ...
    pixel_spacing,sign_IF);

% M =  1.*cos(pd);
% N =  1.*sin(pd);

M =  1.*cos(pd);
N =  1.*sin(pd);

if plot_true

    figure
    quiver( XX, YY, M, N, 0.5, 'k', 'linewidth', 2 );
    set( gca, 'fontname', 'arial', 'fontsize', 14, 'ydir', 'reverse' );
    axis tight
    title('Gradient Vector field of smoothed phasor')
end

% M = smoothn(M,'robust'); %dx
% N = smoothn(N,'robust'); %dy
% mag = smoothn(pm,'robust');
% [XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );

[curl_val] = curl(XX,YY,M,N);
[div_val] = divergence(XX,YY,M,N);


% plot and segment by thresholds
if plot_true
    figure;plot(div_val(:),curl_val(:),'.','MarkerSize',20)
    xlabel('Divergence')
    ylabel('Curl')
    xlim([min([div_val(:); curl_val(:)]) ,max([div_val(:); curl_val(:)])])
    ylim([min([div_val(:); curl_val(:)]) ,max([div_val(:); curl_val(:)])])
    hline(0)
    vline(0)
    title('Div vs Curl to segment eigmap')
    plot_beautify
end

com = [curl_val(:) ;div_val(:)];
m = mean(com);
s = std(com);

curl_val_zscore = (curl_val - m) ./ (s);
curl_val_mask = abs(curl_val_zscore)>1;
curl_val_thresh = curl_val.*curl_val_mask;
if plot_true
    figure;imagesc(curl_val_thresh)
    set(gca,'ydir','reverse')
    plot_beautify
    title('Curl of gradient vector field to detect rotational wave patterns')
    colorbar
end

div_val_zscore = (div_val - m) ./ (s);
div_val_mask = abs(div_val_zscore)>1;
div_val_thresh = div_val.*div_val_mask;
if plot_true
    figure;imagesc(div_val_thresh)
    set(gca,'ydir','reverse')
    plot_beautify
    title('Divergence of gradient vector field to detect expanding wave patterns')
    colorbar
end


% everywhere where curl and div < 0.2 are planar regions
%planar_thresh =( 1-curl_val_thresh) .*(1- div_val_thresh);
%planar_thresh(planar_thresh~=1)=0;

planar_thresh = (abs(div_val_zscore)<0.4) .* (abs(curl_val_zscore)<0.4);
if plot_true
    figure;
    imagesc(planar_thresh)
    plot_beautify
    title('Planar wave regions')
end
CC = bwconncomp(planar_thresh, 4); % 8-connectivity for diagonal neighbors too
planar_regions = cell(CC.NumObjects,1);
num_elem=[];
for ii = 1:CC.NumObjects
    mask_i = false(CC.ImageSize);
    mask_i(CC.PixelIdxList{ii}) = true;
    planar_regions{ii} = mask_i;
    num_elem(ii) = sum(mask_i(:));
end
[aa bb]=max(num_elem);
planar_regions = planar_regions(bb);

% circular linear correlation to get rotation strength
idx = 1:7;
idy=1:6;
% idx = 1:4;
% idy = 1:4;
XX1=XX(idx,idy);
YY1=YY(idx,idy);
M1 = M(idx,idy);
N1 = N(idx,idy);
cl = curl_val(idx,idy);
pl = angle(xph(idx,idy));
d = div_val(idx,idy);
[cc1,pv,center_point] = phase_correlation_rotation( pl, cl,[],sign_IF);
[cc1d,pv,source] = phase_correlation_distance( pl, [],d, 1);

idx = 1:7;
idy=11:16;
% idx = 1:4;
% idy = 1:4;
XX1=XX(idx,idy);
YY1=YY(idx,idy);
M1 = M(idx,idy);
N1 = N(idx,idy);
cl = curl_val(idx,idy);
pl = angle(xph(idx,idy));
d = div_val(idx,idy);
[cc2,pv,center_point] = phase_correlation_rotation( pl, cl,[],sign_IF);
[cc2d,pv,source] = phase_correlation_distance( pl, [],d, 1);

corr_cl=(abs(cc1)+abs(cc2))/2;
%corr_cl(i)=(abs(cc2));
corr_div_cl = cc2d;

%look at planar wave strength in grid
mask = planar_regions{1};
[rho,~] = compute_planar_wave_mask(xphs,mask,XX,YY);
planar_cl = rho;

out.corr_curl = corr_cl;
out.corr_div = corr_div_cl;
out.corr_planar = planar_cl;



