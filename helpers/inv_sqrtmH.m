function A = inv_sqrtmH( B )
% 
[V,D] = eig(B);
d = diag(D);
d = 1./sqrt(d);
A = V*diag(d)*V';