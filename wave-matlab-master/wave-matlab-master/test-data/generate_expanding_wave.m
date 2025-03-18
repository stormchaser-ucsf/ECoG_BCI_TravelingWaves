function [xf] = generate_expanding_wave( image_size, dt, T, ...
                                        freq )
%function [xf] = generate_expanding_wave( image_size, dt, T, ...
%                                        freq )

% Define grid
Nx = image_size; % Number of points in x-direction
Ny = image_size; % Number of points in y-direction
x = linspace(-10, 10, Nx);
y = linspace(-10, 10, Ny);
[X, Y] = meshgrid(x, y);

% Wave properties
%c = 0.5; % Wave speed c = lambda*f
k = 0.5; % Wavenumber related to wavelength lambda by  k =2*pi/lambda
w = freq; % Angular frequency w = 2*pi*f


% Time parameters
dt = dt;
t_max = T;
t = 0:dt:t_max;

% Create figure
%figure;
%colormap(parula);
xf=[];
for i = 1:length(t)
    % Compute radial distance
    R = sqrt(X.^2 + Y.^2);
    
    % Compute wave function
    Z = sin(-k * R - w * t(i));

    % store
    xf(:,:,i) = Z;
    
% %     Plot wave
%     surf(X, Y, Z, 'EdgeColor', 'none');
%     imagesc(Z)
%     axis([-10 10 -10 10 -1 1]);
%     caxis([-1 1]);
%     shading interp;
%     view(2);
%     colorbar;
%     title(['Expanding 2D Wave at t = ', num2str(t(i)), ' s']);
%     drawnow;
end



