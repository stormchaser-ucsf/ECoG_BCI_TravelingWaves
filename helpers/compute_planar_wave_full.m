function [rho,pval,alp_hat] = compute_planar_wave_full(mini_grid,elec)
%function [rho,pval,alp_hat] = compute_planar_wave_full(xph,elec)


pred=elec;
tmp = angle(mini_grid);

% init of reg grid search parameters
alp_range = 0:3:360;
r_range = [(0.01:1:60.0) 60]'; %60 is spatial nyquist limit for electrode spacing of 3mm

theta = tmp;
theta = wrapTo2Pi(theta);
rval=[];

% vectorizing
as = r_range * cosd(alp_range);
len = size(as);
as=as(:);
bs = r_range * sind(alp_range);
bs = bs(:);

theta_hat = pred * ([as';bs']);
y = repmat(theta,1,size(theta_hat,2)) - theta_hat;
r1 = mean(cos(y));
r2 = mean(sin(y));
rtmp = (r1.^2 + r2.^2).^(0.5);
rval = reshape(rtmp,[len]);

% get the best regression parameters
[aa bb]=find(rval==max(rval(:)));
if length(aa)>1
    aa=aa(1);
    bb=bb(1);
end
alp_hat=alp_range;
r_hat=  r_range;
alp_hat = alp_hat(bb);
r_hat = r_hat(aa);
a = r_hat*cosd(alp_hat);
b = r_hat*sind(alp_hat);

% get the phase offset
theta_hat = wrapTo2Pi(pred*([a;b]));
y1 = sum(sin(theta-theta_hat));
y2 = sum(cos(theta-theta_hat));
phi = atan2(y1,y2);

% final reconstruction
theta_hat = wrapTo2Pi(theta_hat + phi);

% get circular correlation
[rho, pval] = circ_corrcc(theta_hat(:), theta(:));

