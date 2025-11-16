function [curl_val,M,N,XX,YY,xphs] = get_curl(xph)

%
% [pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN( xph, ...
%     1,-1);
%
% [XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );
%
% ph=pd;
% M =  1.*cos(ph);
% N =  1.*sin(ph);
% M = smoothn(M,'robust');
% N = smoothn(N,'robust');
%
% %[curl_val]= curl(XX,YY,M,N);
% [curl_val]= divergence(XX,YY,M,N);


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
[curl_val] = curl(XX,YY,M,N);
% interpolate to grid size
%curl_val = imresize(curl_val0,[11 23],'bilinear');
%curl_val = abs(curl_val);




end