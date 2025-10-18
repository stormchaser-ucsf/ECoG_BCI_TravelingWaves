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

% correlation should not drop more than 10%. or just sig. pval?
out = detect_curlVortices_dataDriven(curl_val0,xph, 0.75);

% cc1=mean(corr_val);
% out.curl_val0 = curl_val0;
% out.curl_val = curl_val;
% out.cc1=cc1;
% out.pv=pv;
% out.center_point=center_point;
% out.max_curl = max(abs(curl_val0(:)));

end