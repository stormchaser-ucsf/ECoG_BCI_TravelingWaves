function planar_val = planar_stats_full(xph,grid_layout,elecmatrix)

planar_val=[];k=1;
rows=size(xph,1);
cols=size(xph,2);

radius = 2;% 2 electrodes on either side for a 5 by 5 grid 
for i=1:(rows)
    for j=1:(cols)
        r_min = max(1, i - radius);
        r_max = min(rows, i + radius);
        c_min = max(1, j - radius);
        c_max = min(cols, j + radius);

        % mini grid
        mini_grid = xph(r_min:r_max,c_min:c_max);
        mini_grid = mini_grid(:);

        % % elec locations
        % elec = grid_layout(r_min:r_max,c_min:c_max);
        % elec = elecmatrix(elec(:),:);
        % [c,s,l] = pca(elec);
        % elec = [s(:,1) s(:,2)];
        % 
        % % get planr wave stats for minigrid (r and preferred phase)        
        % [rho,pval,alp] = compute_planar_wave_full(mini_grid,elec);

        % using just the index locations
        [XX,YY] = meshgrid( 1:size(mini_grid,2), 1:size(mini_grid,1) );
        [rho,pval,alp] = compute_planar_wave(mini_grid,XX,YY);

        
        planar_val(i,j) = rho*(cosd(alp) + 1i*sind(alp));


    end
end

end

