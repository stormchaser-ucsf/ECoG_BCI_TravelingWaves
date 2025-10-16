function  [bbox] = expand_bound_box(bbox,cmin,cmax,rmin,rmax,...
            row_flag,col_flag,symm_flag,rows,cols)



% check if possible to expand symmetrically
if cmin == 1 || cmax == cols || rmin == 1 || rmax == rows
    symm_flag=false;
    bbox.symm.flag=symm_flag;

else
    cmin=cmin-1;
    cmax=cmax+1;
    rmin=rmin-1;
    rmax=rmax+1;
    symm_flag = true;
    row_flag=false;
    col_flag=false;
    bbox.symm.flag=symm_flag;
    bbox.symm.cols = [cmin cmax];
    bbox.symm.rows = [rmin rmax];
    bbox.row.flag=row_flag;
    bbox.col.flag=col_flag;
end

% check if possible to expand along the columns
if symm_flag == false
    if cmin == 1 || cmax == cols
        col_flag=false;
        bbox.col.flag=col_flag;
    else
        cmin=cmin-1;
        cmax=cmax+1;
        col_flag=true;
        bbox.col.flag=col_flag;
        bbox.col.cols = [cmin cmax];
        bbox.col.rows = [rmin rmax];
    end

    if rmin == 1 || rmax == rows
        row_flag=false;
        bbox.row.flag=row_flag;
    else
        rmin=rmin-1;
        rmax=rmax+1;
        row_flag=true;
        bbox.row.flag=row_flag;
        bbox.row.cols = [cmin cmax];
        bbox.row.rows = [rmin rmax];
    end



end

