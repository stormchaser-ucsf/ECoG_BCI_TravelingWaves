function [hh,fg,cb] = plot_vector_field_NN( ph, pm,varargin )
% *WAVE*
%
% PLOT VECTOR FIELD
%
% INPUT
% ph - complex-valued scalar field representing vector directions
%
% OUTPUT
% vector field plot
%

% set defaults
if nargin > 2, plot_option = varargin{1}; else plot_option = 0; end

% init
[XX,YY] = meshgrid( 1:size(ph,2), 1:size(ph,1) );

M =  pm.*cos(ph);
N =  pm.*sin(ph);
% 
% M =  1.*cos(ph);
% N =  1.*sin(ph);
% 
%M = pm .* smoothn(M,'robust');
%N = pm .* smoothn(N,'robust');

M = smoothn(M,'robust');
N = smoothn(N,'robust');

% plotting
fg = figure;
if ( plot_option == 0 )
    imagesc( (ph) ); cb = colorbar; axis image; caxis( [-pi pi] ); hold on;
    set( get(cb,'ylabel'), 'string', 'Direction (rad)' )
    quiver( XX, YY, M, N, 0.25, 'r' );
    set( gca, 'fontname', 'arial', 'fontsize', 14, 'ydir', 'reverse' ); hh = gca;
elseif ( plot_option == 1 )
    ih = imagesc( (ph) ); cb = colorbar; 
    axis image; 
    %clim( [-pi pi] ); 
    hold on;
    set( get(cb,'ylabel'), 'string', 'Direction (rad)' )
    quiver( XX, YY, M, N, 0.5, 'k', 'linewidth', 2 );
    set( gca, 'fontname', 'arial', 'fontsize', 14, 'ydir', 'reverse' ); hh = gca;
    delete( ih ); delete( cb ); axis off
end

% % plot curl also (or divergence)
[d,c]= curl(XX,YY,M,N);
% [d]= divergence(XX,YY,M,N);
hold on
contour(XX,YY,d,'ShowText','on','LineWidth',1)
% % figure;imagesc(d)
