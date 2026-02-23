function [res,bad_ch] = get_PCs_VAF_DimCollapse(stats_cl_hg_days,num_targets)


%%%% EXTRA STUFF
%%%% METHOD OF PRINICIPAL ANGLES AND
for tid=1:num_targets
    act_wave=[];
    act_nonwave=[];
    bad_ch=[];
    for days=1:length(stats_cl_hg_days)-1
        stats_cl_hg = stats_cl_hg_days{days};
        dim_wave=[];
        dim_nonwave=[];
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
                %tmp = tmp-mean(tmp,1);
                act_wave = [act_wave;tmp];

                tmp = stats_cl_hg(i).hg_nonwave;
                tmp = cell2mat(tmp');
                %tmp = tmp-mean(tmp,1);
                act_nonwave = [act_nonwave;tmp];
            end
        end
    end

    s = std(act_wave,1);
    s = zscore((s));
    xx = find(abs(s)>5);
    bad_ch =[bad_ch xx];

    s = std(act_nonwave,1);
    s = zscore((s));
    xx = find(abs(s)>5);
    bad_ch =[bad_ch xx];
    xx = unique(bad_ch);

    act_wave(:,xx) = 1e-20*randn(size(act_wave,1),length(xx));
    act_nonwave(:,xx) = 1e-20*randn(size(act_nonwave,1),length(xx));
    
    dataTensor=[];
    if size(act_nonwave,1) > size(act_wave,1)
        l = size(act_nonwave,1) - size(act_wave,1);
        k = randperm(l,1);
        k1 = k + size(act_wave,1)-1;
        act_nonwave1 = act_nonwave(k:k1,:);
        act_nonwave1 = act_nonwave1 - mean(act_nonwave1);
        act_wave = act_wave - mean(act_wave);
        dataTensor(:,:,1) = act_wave;
        dataTensor(:,:,2) = act_nonwave1;
    end
    
    prin_angles = compute_prin_angles_PC(dataTensor,50);
    figure;plot(prin_angles)
    hold on
    maxEntropy = run_tme(dataTensor,'surrogate-TC');
    prin_angles_boot=[];
    addpath(genpath('/home/user/Documents/Documents/GitHub/ManifoldAnalysisPCA/TME/'))
    parfor inner_loop=1:200   
        surrTensor = simulate_time(maxEntropy);
        tmp_a = compute_prin_angles_PC(surrTensor,50);
        prin_angles_boot(inner_loop,:) = tmp_a;
    end
    prin_angles_boot = sort(prin_angles_boot);
    plot(prin_angles_boot(1,:))


    % DO PCA and find how much VAF each PC captures in either wave or nonwave    
    [c,s,l] = pca(act_nonwave1);
    % for nonwave
    R=[];
    parfor ii=1:size(act_nonwave1,2)
        tmp = act_nonwave1 - act_nonwave1*c(:,ii)*c(:,ii)';
        num= norm(tmp,'fro')^2;
        den = norm(act_nonwave1,'fro')^2;
        R(ii) = 1 - num/den;
    end

    R1=[];
    parfor ii=1:size(act_wave,2)
        tmp = act_wave - act_wave*c(:,ii)*c(:,ii)';
        num= norm(tmp,'fro')^2;
        den = norm(act_wave,'fro')^2;
        R1(ii) = 1 - num/den;
    end
    figure;stem(R)
    hold on
    stem(R1)

    %%%%% parcellate wave and nonwave unique sources of variance
    [c,s,l] = pca(act_wave);
    [c1,s1,l1] = pca(act_nonwave1);
    % take the top 100PCs and do SVD to get common subspace
    W = [c(:,1:100) c1(:,1:100)];
    [U,s,v]= svd(W,0);

    % project data onto common space
    Lwave = act_wave*U;
    Lnonwave = act_nonwave1*U;

    % get nonwave-specific subspace
    [c,s,l]=pca(Lwave); % take the bottom 20PCs
    Uwave_null = c(:,181:end);
    Proj = Lnonwave*Uwave_null;
    [c,s,l] = pca(Proj); % keep all the PCs at this point
    Vnw_wavenull = c;
    Z_nwunique = Uwave_null*Vnw_wavenull;

    % get the percentage variance captured by nonwave specific subspace
    Ywave  = Lwave*Z_nwunique;
    Ynonwave  = Lnonwave*Z_nwunique;

    % compute VAF
    R=[];
    for ii=1:size(Ywave,2)
        tmp = Lwave - Lwave*Z_nwunique(:,ii)*Z_nwunique(:,ii)';
        num=norm(Lwave,'fro')^2 - norm(tmp,'fro')^2;
        den = norm(Lwave,'fro')^2;
        R(ii) = num/den;
    end
    figure;stem(R*100)
end





% if lengths differ, then computer the average covariance matrix for length
% matched, along with TME percentiles. 

res=[];res_var=[];
bad_ch=[];
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
        bad_ch =[bad_ch xx];
        act_wave(:,xx) = 1e-4*randn(size(act_wave,1),length(xx));

        %[c,s,l]=pca(zscore(act_wave));l0=l;c0=c;
        % dimensionality
        
        if size(act_wave,1) > size(act_nonwave,1)
            %idx=randperm(size(act_wave,1),size(act_nonwave,1));
            %idx =1:size(act_nonwave,1);
            %[c,s,l]=pca((act_nonwave(idx,:)));
            l = size(act_wave,1) - size(act_nonwave,1);
            idx  = l+1:size(act_wave,1) ;
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
        bad_ch =[bad_ch xx];
        act_nonwave(:,xx) = 1e-4*randn(size(act_nonwave,1),length(xx));

        % dimensionality        
        if size(act_wave,1) < size(act_nonwave,1)
            %idx=randperm(size(act_nonwave,1),size(act_wave,1));
            %[c,s,l]=pca((act_nonwave(idx,:)));
            %[c,s,l]=pca(zscore(act_nonwave(idx,:)));
            %idx =1:size(act_wave,1);
            l = size(act_nonwave,1) - size(act_wave,1);
            idx  = l+1:size(act_nonwave,1) ;
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

        if wave_len_cl(i)<nonwave_len_cl(i)
            wave_plv(i,:) = (angle(mean(a)));

            len = min(30,nonwave_len_cl(i) -  wave_len_cl(i));
            idx=randperm(nonwave_len_cl(i) -  wave_len_cl(i),len);
            plv_tmp=[];
            for j=1:length(idx)
                tmp = b(idx(j):idx(j)+wave_len_cl(i)-1,:);
                plv_tmp(j,:) = (angle(mean(tmp,1)));
            end
            nonwave_plv(i,:) = circ_mean(plv_tmp);

        elseif wave_len_cl(i)>nonwave_len_cl(i)
            nonwave_plv(i,:) = (angle(mean(b)));

            len = min(30,wave_len_cl(i) -  nonwave_len_cl(i));
            idx=randperm(wave_len_cl(i) -  nonwave_len_cl(i),len);
            plv_tmp=[];
            for j=1:length(idx)
                tmp = a(idx(j):idx(j)+nonwave_len_cl(i)-1,:);
                plv_tmp(j,:) = (angle(mean(tmp,1)));
            end
            wave_plv(i,:) = circ_mean(plv_tmp);

        elseif wave_len_cl(i)== nonwave_len_cl(i)
            nonwave_plv(i,:) = (angle(mean(b)));
            wave_plv(i,:) = (angle(mean(a)));
        end



        dim_wave=[dim_wave pr_wave];
        dim_nonwave=[dim_nonwave pr_nonwave];
    end
    res=[res;[mean(dim_wave) mean(dim_nonwave)]];
end
%res
% dimensionality
%[p,h]=signrank(res(:,1),res(:,2))


