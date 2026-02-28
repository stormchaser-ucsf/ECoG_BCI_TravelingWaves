function plot_on_brain(ch_wts,cortex,elecmatrix,ecog_grid)


% plot PAC on brain
ch_layout=[];
for i=1:23:253
    ch_layout = [ch_layout; i:i+22 ];
end
ch_layout = (fliplr(ch_layout));
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix, 'color', 'w','msize',2);
for j=1:256%length(sig)
    [xx yy]=find(ecog_grid==j);
    if ~isempty(xx)
        ch = ch_layout(xx,yy);
        ms = ch_wts(j)*7;
        
        e_h = el_add(elecmatrix(ch,:), 'color', 'b','msize',ms);
    end
end
set(gcf,'Color','w')
