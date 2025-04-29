function plot_on_brain_B1_253(ch_wts,cortex,elecmatrix,ecog_grid,weighting)


% elecmatrix already arranged as per BR-NSP conversion

% % plot PAC on brain
% ch_layout=[];
% for i=1:23:253
%     ch_layout = [ch_layout; i:i+22 ];
% end
% ch_layout = (fliplr(ch_layout));
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix, 'color', 'w','msize',2);
for j=1:253%length(sig)        
    
    ms = ch_wts(j)*weighting;
    e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',ms);

end
set(gcf,'Color','w')
