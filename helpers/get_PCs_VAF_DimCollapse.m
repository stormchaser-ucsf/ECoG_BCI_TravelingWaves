function [res] = get_PCs_VAF_DimCollapse(stats_cl_hg_days,num_targets)


res=[];res_var=[];
for days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    dim_wave=[];
    dim_nonwave=[];
    for tid=1:num_targets
        act_wave=[];
        act_nonwave=[];
        for i=1:length(stats_cl_hg)
            if stats_cl_hg(i).target_id ==tid

                tmp = stats_cl_hg(i).hg_wave;
                % tmp_data=[];
                % for j=1:length(tmp)
                %     a=tmp{j};
                %     a = a-mean(a);
                %     tmp_data=[tmp_data;a];
                % end
                tmp = cell2mat(tmp');
                tmp = tmp-mean(tmp,1);
                act_wave = [act_wave;tmp];

                tmp = stats_cl_hg(i).hg_nonwave;
                tmp = cell2mat(tmp');
                tmp = tmp-mean(tmp,1);
                act_nonwave = [act_nonwave;tmp];
            end
        end


        % find and reject bad channels 
        s = std(act_wave,1);
        s = zscore((s));
        xx = find(abs(s)>5);
        act_wave(:,xx) = 1e-4*randn(size(act_wave,1),length(xx));

        %[c,s,l]=pca(zscore(act_wave));l0=l;c0=c;
        % dimensionality
        
        if size(act_wave,1) > size(act_nonwave,1)
            idx=randperm(size(act_wave,1),size(act_nonwave,1));
            %[c,s,l]=pca((act_nonwave(idx,:)));
            [c,s,l]=pca(zscore(act_wave(idx,:)));
        else
            [c,s,l]=pca(zscore(act_wave(:,:)));
        end
        %pr_wave = ((sum(l))^2) / (sum(l.^2));
        vaf = cumsum(l)./sum(l);
        [aa bb]=find(vaf>0.8);
        pr_wave = aa(1); % z score data matrix

        % total variance
        %pr_wave=sum(l)
        %pr_wave = sum(log(l(1:5))); % dont z-score data matrix

        %%%% nonwave epochs analyses
        % find and reject bad channels 
        s = std(act_nonwave,1);
        s = zscore((s));
        xx = find(abs(s)>5);
        act_nonwave(:,xx) = 1e-4*randn(size(act_nonwave,1),length(xx));

        % dimensionality        
        if size(act_wave,1) < size(act_nonwave,1)
            idx=randperm(size(act_nonwave,1),size(act_wave,1));
            %[c,s,l]=pca((act_nonwave(idx,:)));
            [c,s,l]=pca(zscore(act_nonwave(idx,:)));
        else
            [c,s,l]=pca(zscore(act_nonwave(:,:)));
        end
        %pr_nonwave = ((sum(l))^2) / (sum(l.^2));
        vaf = cumsum(l)./sum(l);
        [aa bb]=find(vaf>0.8);
        pr_nonwave = aa(1); % z score data matrix

        % total variance
        %pr_nonwave = sum(l); % dont z-score data matrix
        %pr_nonwave = sum(log(l(1:5))); % dont z-score data matrix


        dim_wave=[dim_wave pr_wave];
        dim_nonwave=[dim_nonwave pr_nonwave];
    end
    res=[res;[median(dim_wave) median(dim_nonwave)]];
end
%res
% dimensionality
%[p,h]=signrank(res(:,1),res(:,2))


