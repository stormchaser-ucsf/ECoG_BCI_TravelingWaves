function planar_val = planar_stats_muller(xph)

M = real(xph);
N = imag(xph);
tmp = smoothn({M,N},'robust');
M = tmp{1}; N = tmp{2};
xphs = M+1j*N;


[pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN( xphs, ...
    1,-1);

ph=pd;
M =  pm.*cos(ph);
N =  pm.*sin(ph);
tmp = smoothn({M,N},'robust');
M = tmp{1}; N = tmp{2};
planar_val = M + 1j*N;

% 
% M = real(planar_val);
% N = imag(planar_val);
% [XX,YY] = meshgrid( 1:size(planar_val,2), 1:size(planar_val,1) );
% figure;
% quiver(XX,YY,M,N);axis tight

