
%% STEP 1: SIMULATE A 2D TRAVELING WAVE 

close all;
clear;
clc

% Parameters
Lx = 10;          % Length in x direction
Ly = 10;          % Length in y direction
Nx = 100;         % Number of points in x
Ny = 100;         % Number of points in y
c = 2.5;            % Wave speed
T = 5;            % Total time
dt = 0.05;         % Time step

% Create spatial grid
x = linspace(0, Lx, Nx);
y = linspace(0, Ly, Ny);
[X, Y] = meshgrid(x, y);
data=[];
tt=[];
v = VideoWriter('myVideo.avi');
v.FrameRate=6;
open(v);
% Time loop
for t = 0:dt:T
    % Calculate the wave function
    wave = sin(2 * pi * (0.25*X + 0.25*Y + c * t) );
    wave = wave + 0.05*randn(size(wave));
    data = cat(3,data,wave);
    tt=[tt t];
    
    
    % Plotting
    %surf(X, Y, wave, 'EdgeColor', 'none');
    %axis([0 Lx 0 Ly -1 1]); % Set axis limits
    surf(X(1:20,1:20), Y(1:20,1:20), wave(1:20,1:20), 'EdgeColor', 'none');        
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Wave Amplitude');
    title(['Planar Traveling Wave at t = ' num2str(t)]);
    colorbar;
    view(2);  % 2D view
    axis tight
    pause(0.1); % Pause for a moment to visualize
    %frame = im2frame(images{u});
    cc=getframe(gcf);
    writeVideo(v, cc);
end
close(v)

%% STEP 2: DETECT TRAVELING WAVE AND IDENTIFY ITS PROPERTIES

% add the circ stats toolbox
addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')

% compute the fft of the signal 
x=squeeze(data(1,1,:));
Fs=1/dt;
figure;plot(tt,x);
[psdx,ffreq,phasex]=fft_compute(x,Fs,1);

% design band pass filter between 0.8 and 1.2Hz
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',2.35,'HalfPowerFrequency2',2.65, ...
    'SampleRate',Fs);
%fvtool(bpFilt)

% extracting phase time series for the 2D data : important point is to let
% some filter run off happen before getting the phase data
data1 = permute(data,[3 1 2]);
data1 = cat(1,randn(250,100,100),data1);
tmp = filtfilt(bpFilt,data1);
tmp = tmp(251:end,:,:);
tmp1 = angle(hilbert(tmp));
tmp = permute(tmp,[2 3 1]);
tmp1 = permute(tmp1,[2 3 1]);
x1  = squeeze(tmp(1,1,:));
figure;plot(tt,x)
hold on
plot(tt,x1)

phdata=tmp1;
xx=1:size(phdata,1);
yy=1:size(phdata,2);

% creating the predictor values
pred = [];
for i=1:20
    pred = [pred; [ (1:20)' repmat(i,20,1)]];
end
pred(:,1) = pred(:,1)./max(pred(:,1));
pred(:,2) = pred(:,2)./max(pred(:,2));

% performing 2D circular linear correlation at each time-point
for i=1:size(phdata,3)
    tmp = phdata(:,:,i); % this is the 2D phase data across the grid 
    tmp = tmp(1:20,1:20);
    figure;imagesc(tmp)
    
    theta_orig = tmp;


    % iteratively solve for cirular linear regression
    % model is theta_hat = ax + by + e , where x and y are spatial
    % coordinates

    theta = tmp(:);
    theta = wrapTo2Pi(theta);
    rval=[];

    % vectorizing
    alp_range = 0:0.5:360;
    r_range = (0.01:0.05:20)';

    as = r_range * cosd(alp_range);
    bs = r_range * sind(alp_range);
    
    for alp=0:0.5:360
        rtmp=[];
        for r = 0.01:0.05:20
            a = r*cosd(alp);
            b = r*sind(alp);
            theta_hat = pred*([a;b]);    
            theta_hat = wrapTo2Pi(theta_hat);

            y = theta-theta_hat;
            r1 = mean(cos(y));
            r2 = mean(sin(y));
            rtmp = [rtmp ;sqrt(r1^2 + r2^2)];
        end
        rval = [rval rtmp];
    end

    % get the best regression parameters
    [aa bb]=find(rval==max(rval(:)));    
    alp_hat=0:0.5:360;
    r_hat=  0.01:0.05:20;
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
    theta_hat = wrapToPi(theta_hat + phi);

    % rearranging as a 2D array
    theta_hat = reshape(theta_hat,[size(theta_orig)]);
    figure;
    subplot(1,2,1)
    imagesc(theta_orig)
    axis tight
    title('Simulated Original')
    subplot(1,2,2)
    imagesc(theta_hat)
    title('Recon from circular linear regression')
    axis tight

    % get circular correlation 
    [rho pval] = circ_corrcc(theta_hat(:), theta_orig(:));
    disp(rho)
    
end

%% STEP 2A: DETECT TRAVELING WAVE AND IDENTIFY ITS PROPERTIES
% vectorized code for speed 

% add the circ stats toolbox
addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')

% compute the fft of the signal 
x=squeeze(data(1,1,:));
Fs=1/dt;
figure;plot(tt,x);
[psdx,ffreq,phasex]=fft_compute(x,Fs,1);

% design band pass filter between 0.8 and 1.2Hz
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',2.35,'HalfPowerFrequency2',2.65, ...
    'SampleRate',Fs);
%fvtool(bpFilt)

% extracting phase time series for the 2D data : important point is to let
% some filter run off happen before getting the phase data
data1 = permute(data,[3 1 2]);
data1 = cat(1,randn(250,100,100),data1);
tmp = filtfilt(bpFilt,data1);
tmp = tmp(251:end,:,:);
tmp1 = angle(hilbert(tmp));
tmp = permute(tmp,[2 3 1]);
tmp1 = permute(tmp1,[2 3 1]);
x1  = squeeze(tmp(1,1,:));
figure;plot(tt,x)
hold on
plot(tt,x1)

phdata=tmp1;
xx=1:size(phdata,1);
yy=1:size(phdata,2);

% creating the predictor values
pred = [];
for i=1:20
    pred = [pred; [ (1:20)' repmat(i,20,1)]];
end
pred(:,1) = pred(:,1)./max(pred(:,1));
pred(:,2) = pred(:,2)./max(pred(:,2));

% performing 2D circular linear correlation at each time-point
for i=1:size(phdata,3)
    tmp = phdata(:,:,i); % this is the 2D phase data across the grid 
    tmp = tmp(1:20,1:20);
    figure;imagesc(tmp)
    
    theta_orig = tmp;


    % iteratively solve for cirular linear regression
    % model is theta_hat = ax + by + e , where x and y are spatial
    % coordinates

    theta = tmp(:);
    theta = wrapTo2Pi(theta);
    rval=[];

    % vectorizing
    alp_range = 0:0.5:360;
    r_range = (0.01:0.02:25)';

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
    alp_hat=0:0.5:360;
    r_hat=  0.01:0.05:20;
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
    theta_hat = wrapToPi(theta_hat + phi);

    % rearranging as a 2D array
    theta_hat = reshape(theta_hat,[size(theta_orig)]);
    figure;
    subplot(1,2,1)
    imagesc(theta_orig)
    axis tight
    title('Simulated Original')
    subplot(1,2,2)
    imagesc(theta_hat)
    title('Recon from circular linear regression')
    axis tight

    % get circular correlation 
    [rho pval] = circ_corrcc(theta_hat(:), theta_orig(:));
    disp(rho)
    
end


% identification of stable epochs


%%
% Parameters
R = 5;           % Maximum radius of the wave
N = 200;         % Number of spatial points (resolution)
c = 1;           % Wave speed
T = 10;          % Total time
dt = 0.1;        % Time step
time_steps = T/dt;

% Create spatial grid in Cartesian coordinates
x = linspace(-R, R, N);
y = linspace(-R, R, N);
[X, Y] = meshgrid(x, y);

% Iterate over time
for t = 0:dt:T
    % Calculate the radius from the center for each point
    radius = sqrt(X.^2 + Y.^2);
    
    % Define the wave function (traveling wave)
    wave = sin(2 * pi * (radius - c * t));  % Traveling wave solution
    
    % Plotting
    surf(X, Y, wave, 'EdgeColor', 'none');
    axis equal;
    axis([-R R -R R -1 1]); % Set axis limits
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Wave Amplitude');
    title(['Circular Planar Traveling Wave at t = ' num2str(t)]);
    colorbar;
    view(2);  % 2D view
    pause(0.1); % Pause for visualization
end