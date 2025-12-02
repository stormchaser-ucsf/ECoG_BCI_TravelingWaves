function [planar_val,planar_reg] = planar_stats_full(xph,grid_layout,elecmatrix)


% init for grid search
alp_range = 0:3:360;
r_range = [(0.01:1:60.0) 60]'; %60 is spatial nyquist limit for electrode spacing of 3mm
alp_range = deg2rad(alp_range);

planar_val=[];k=1;
rows=size(xph,1);
cols=size(xph,2);
planar_reg=[];
radius = 5;% 2 electrodes on either side for a 5 by 5 grid 
for i=1:(rows)
    for j=1:(cols)
        r_min = max(1, i - radius);
        r_max = min(rows, i + radius);
        c_min = max(1, j - radius);
        c_max = min(cols, j + radius);

        % mini grid
        mini_grid = xph(r_min:r_max,c_min:c_max);
        mini_grid = mini_grid(:);

        % % get planr wave stats for minigrid (r and preferred phase) using       
        % % elec locations
        elec = grid_layout(r_min:r_max,c_min:c_max);
        elec = elecmatrix(elec(:),:);
        [c,s,l] = pca(elec);
        elec = [s(:,1) s(:,2)];
        [rho,pval,alp,rhat] = compute_planar_wave_full(mini_grid,elec);
        % result = circular_linear_regression(elec(:,1),elec(:,2),angle(mini_grid),...
        %     r_range,alp_range);
        % rhat = result.rho;
        % rho = result.R;
        % alp = result.alpha;


        % using just the index locations
        % [XX,YY] = meshgrid( 1:size(mini_grid,2), 1:size(mini_grid,1) );
        % [rho,pval,alp] = compute_planar_wave(mini_grid,XX,YY);

        
        % planar_val(i,j) = rho*(cosd(alp) + 1i*sind(alp));
        % planar_reg(i,j) = rhat*(cosd(alp) + 1i*sind(alp));
        
        planar_val(i,j) = rho*(cos(alp) + 1i*sin(alp));
        planar_reg(i,j) = rhat*(cos(alp) + 1i*sin(alp));



    end
end

end

