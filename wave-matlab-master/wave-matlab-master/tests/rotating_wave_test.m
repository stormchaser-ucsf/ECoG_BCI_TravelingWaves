% *WAVE*
%
% PHASE GRADIENT TEST SCRIPT     takes the phase gradient of a test
%                                   test signal (here, a 2d target wave)
%

%addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\wave-matlab-master\wave-matlab-master'))

clear all; clc; close all 
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves'))

% parameters
T = 10; %s
Fs = 100; freq = 10; %Hz
image_size = 30; %px
pixel_spacing = 1; %a.u.
direction = +1; % +1/-1
dt=1/Fs;

% generate data
%xf = generate_rotating_wave( image_size, 1/Fs, T, freq, direction );
xf=generate_expanding_wave( image_size, dt, T, freq);

%plot it
figure;
for i=1:size(xf,3)
    imagesc(squeeze(xf(:,:,i)));
    colormap bone    
    pause(0.01)    
end

% z-score data
xf = zscore_independent( xf );

% form analytic signal
xph = analytic_signal( xf );

% calculate instantaneous frequency 
[wt,signIF] = instantaneous_frequency( xph, Fs );

% calculate phase gradient
[pm,pd,dx,dy] = phase_gradient_complex_multiplication( xph, pixel_spacing, signIF );

% using gradient

% plot resulting vector field
plot_vector_field( exp( 1i .* pd(:,:,10) ), 1 );

[XX,YY] = meshgrid( 1:size(pd,2), 1:size(pd,1) );

% expanding wave metrics
for i=1:10%size(pd,3)
    plot_vector_field( exp( 1i .* pd(:,:,i) ), 1 );
    tmp=exp( 1i .* pd(:,:,i) );
    M = real( exp( 1i * angle(tmp) ) ); N = imag( exp( 1i * angle(tmp) ) );
    M = smoothn(M);
    N = smoothn(N);
    [cl,c]= curl(XX,YY,M,N);
    pl = squeeze(angle(xph(:,:,i)));
    [cc,pv,center_point] = phase_correlation_rotation( pl, cl,[],signIF);
    title(['Correlation of ' num2str(cc)]);
end


[cc,pv] = phase_correlation_distance( pl, source, pixel_spacing );


% % plot movie
figure;
play_movie(xf)
