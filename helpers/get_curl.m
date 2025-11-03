function [curl_val,M,N,XX,YY] = get_curl(xph)


[pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN( xph, ...
    1,-1);

[XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );

ph=pd;
M =  1.*cos(ph);
N =  1.*sin(ph);
M = smoothn(M,'robust');
N = smoothn(N,'robust');

[curl_val]= curl(XX,YY,M,N);
%[curl_val]= divergence(XX,YY,M,N);




end