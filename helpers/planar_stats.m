function planar_val = planar_stats(xph)

rows = [1 5];
cols = [1:4:17];
planar_val={};k=1;
for i=1:1%length(rows)
    r = [rows(i):rows(i)+6];
    for j=1:1%length(cols)
        c = [cols(j):cols(j)+6];
        mini_grid = xph(r,c);

        % get planr wave stats for minigrid (r and preferred phase)
        [XX,YY] = meshgrid( 1:size(mini_grid,2), 1:size(mini_grid,1) );
        [rho,pval] = compute_planar_wave(mini_grid,XX,YY);

        planar_val(k).grid_rows=r;
        planar_val(k).grid_cols=c;
        planar_val(k).rho=rho;
        planar_val(k).pval=pval;
        k=k+1;
    end
end

end

