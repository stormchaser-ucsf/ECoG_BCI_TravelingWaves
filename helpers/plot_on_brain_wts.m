function plot_on_brain_wts(tmp,tmp_pac,ecog_grid,cortex,elecmatrix)
%function plot_on_brain_wts(tmp,ecog_grid)
% plot PLV on sig. channels 


ch_wts = [tmp(1:107) 0 tmp(108:111) 0  tmp(112:115) 0 ...
    tmp(116:end)];

% figure;
% imagesc(ch_wts(ecog_grid))

ch_pac = [tmp_pac(1:107) 0 tmp_pac(108:111) 0  tmp_pac(112:115) 0 ...
    tmp_pac(116:end)];

figure;
imagesc(ch_wts(ecog_grid) .* ch_pac(ecog_grid))


ch_layout=[];
for i=1:23:253
    ch_layout = [ch_layout; i:i+22 ];
end
ch_layout = (fliplr(ch_layout));


figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix, 'color', 'w','msize',2);
for j=1:256
    [xx yy]=find(ecog_grid==j);
    if ~isempty(xx) && ch_wts(j)>0
        ch = ch_layout(xx,yy);
        ms = ch_pac(j)*8;
        e_h = el_add(elecmatrix(ch,:), 'color', 'b','msize',ms);
    end
end
set(gcf,'Color','w')

end