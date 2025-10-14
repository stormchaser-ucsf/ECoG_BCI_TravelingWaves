function out=curl_stats(xph)



[XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );
M = real(xph);
N = imag(xph);
M = smoothn(M,'robust'); %dx
N = smoothn(N,'robust'); %dy
xphs = M + 1j*N; % smoothed phasor field: gets smoothed estimates of phase
[pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN( xphs, ...
    1,-1);
% smooth phasors to get smoothed estimates of phase
M = 1.*cos(pd);
N = 1.*sin(pd);
% get curl
[curl_val0] = curl(XX,YY,M,N);
% interpolate to grid size
curl_val = imresize(curl_val0,[11 23],'bilinear');
curl_val = abs(curl_val);

% get extent of rotations
if max(abs(curl_val0(:))) > 0.75
    %[~, ~, symMask, ~] = rotational_extent_from_curl((curl_val0));
    vortices=detect_nonoverlapping_vortices(curl_val0,0.75,0.25,5);
    [rowIdx, colIdx] = find(symMask==1);
    rmin = min(rowIdx); rmax = max(rowIdx);
    cmin = min(colIdx); cmax = max(colIdx);


    % circular linear regression
    pl = angle(xph(rmin:rmax, cmin:cmax));
    [cc1,pv,center_point] = phase_correlation_rotation( pl,...
        curl_val0(rmin:rmax, cmin:cmax),[],-1);
else
    cc1=0;
    pv=1;
    center_point=[];
end

out.curl_val0 = curl_val0;
out.curl_val = curl_val;
out.cc1=cc1;
out.pv=pv;
out.center_point=center_point;
out.max_curl = max(abs(curl_val0(:)));

end