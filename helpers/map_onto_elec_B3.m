function [wts] = map_onto_elec_B3(indata,ecog_grid,grid_layout)
%function [wts] = map_onto_elec_B3(indata)

tmp = indata;
wts=zeros(253,1);
for i=1:253
    if i<=107
        j=i;
    elseif i>=108 && i<=111
        j=i+1;
    elseif i>=112 && i<=115
        j=i+2;
    elseif i>=116
        j=i+3;
    end
    [aa bb] = find(ecog_grid==j);
    idx = grid_layout(aa,bb);
    wts(idx) = tmp(aa,bb);
end

