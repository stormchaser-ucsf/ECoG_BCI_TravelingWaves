function  [bbox] = expand_bound_box_all(bbox,cmin,cmax,rmin,rmax,...
    rows,cols)



% expand in x minus
if cmin==1
    col_lt_flag = false;
    bbox.collt.flag = col_lt_flag;
else
    bbox.collt.cols = [cmin-1 cmax];
    bbox.collt.rows = [rmin rmax];
    bbox.collt.flag = true;
end

% expand in x pos
if cmax==cols
    col_rt_flag = false;
    bbox.colrt.flag = col_rt_flag;
else
    bbox.colrt.cols = [cmin cmax+1];
    bbox.colrt.rows = [rmin rmax];
    bbox.colrt.flag = true;
end

% expand in y minus
if rmin==1
    row_up_flag = false;
    bbox.rowu.flag = row_up_flag;
else
    bbox.rowu.cols = [cmin cmax];
    bbox.rowu.rows = [rmin-1 rmax];
    bbox.rowu.flag=true;
end

% expand in y pos
if rmax==rows
    row_down_flag = false;
    bbox.rowd.flag = row_down_flag;
else
    bbox.rowd.cols = [cmin cmax];
    bbox.rowd.rows = [rmin rmax+1];
    bbox.rowd.flag=true;
end

end

