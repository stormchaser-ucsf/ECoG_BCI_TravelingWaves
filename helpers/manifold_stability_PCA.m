function [prin_angles_overall,prin_angles_boot_overall] = ...
    manifold_stability_PCA(stats_cl_hg_days,num_targets)
% function [,prin_angles_overall,prin_angles_boot_overall] = ...
%     manifold_stability_PCA(stats_cl_hg_days,num_targets)

addpath(genpath('/home/user/Documents/Documents/GitHub/ManifoldAnalysisPCA/'))
%addpath(genpath('/home/user/Documents/Documents/GitHub/ManifoldAnalysisPCA/TME/'))
prin_angles_boot_overall=[];
prin_angles_overall=[];
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
    for outer_loop=1:10

        if size(act_nonwave,1) > size(act_wave,1)
            l = size(act_nonwave,1) - size(act_wave,1);
            k = randperm(l,1);
            k1 = k + size(act_wave,1)-1;
            act_nonwave1 = act_nonwave(k:k1,:);
            act_nonwave1 = act_nonwave1 - mean(act_nonwave1);
            act_wave = act_wave - mean(act_wave);
            dataTensor(outer_loop,:,:,1) = act_wave;
            dataTensor(outer_loop,:,:,2) = act_nonwave1;
        elseif size(act_nonwave,1) < size(act_wave,1)
            l = - size(act_nonwave,1) + size(act_wave,1);
            k = randperm(l,1);
            k1 = k + size(act_nonwave,1)-1;
            act_wave1 = act_wave(k:k1,:);
            act_wave1 = act_wave1 - mean(act_wave1);
            act_nonwave = act_nonwave - mean(act_nonwave);
            dataTensor(outer_loop,:,:,1) = act_wave1;
            dataTensor(outer_loop,:,:,2) = act_nonwave;
        end
    end
    prin_angles_boot=[];
    prin_angles_tmp=[];    
    for loop = 1:size(dataTensor,1)
        disp(loop)
        dataTensor1 = squeeze(dataTensor(loop,:,:,:));
        prin_angles = compute_prin_angles_PC(dataTensor1,50);
        prin_angles_tmp(loop,:) = prin_angles;
        %figure;plot(prin_angles)
        %hold on
        maxEntropy = run_tme(dataTensor1,'surrogate-TC');
        prin_angles_boot_tmp=[];
        parfor inner_loop=1:100
            surrTensor = simulate_time(maxEntropy);
            tmp_a = compute_prin_angles_PC(surrTensor,50);
            prin_angles_boot_tmp(inner_loop,:) = tmp_a;
        end
        prin_angles_boot = cat(1,prin_angles_boot,prin_angles_boot_tmp);
    end

    prin_angles_boot_overall(tid,:,:) = prin_angles_boot';
    prin_angles_overall(tid,:) = mean(prin_angles_tmp,1);

end




%
%
%     % DO PCA and find how much VAF each PC captures in either wave or nonwave
%     [c,s,l] = pca(act_nonwave1);
%     % for nonwave
%     R=[];
%     parfor ii=1:size(act_nonwave1,2)
%         tmp = act_nonwave1 - act_nonwave1*c(:,ii)*c(:,ii)';
%         num= norm(tmp,'fro')^2;
%         den = norm(act_nonwave1,'fro')^2;
%         R(ii) = 1 - num/den;
%     end
%
%     R1=[];
%     parfor ii=1:size(act_wave,2)
%         tmp = act_wave - act_wave*c(:,ii)*c(:,ii)';
%         num= norm(tmp,'fro')^2;
%         den = norm(act_wave,'fro')^2;
%         R1(ii) = 1 - num/den;
%     end
%     figure;stem(R)
%     hold on
%     stem(R1)
%
%     %%%%% parcellate wave and nonwave unique sources of variance
%     [c,s,l] = pca(act_wave);
%     [c1,s1,l1] = pca(act_nonwave1);
%     % take the top 100PCs and do SVD to get common subspace
%     W = [c(:,1:100) c1(:,1:100)];
%     [U,s,v]= svd(W,0);
%
%     % project data onto common space
%     Lwave = act_wave*U;
%     Lnonwave = act_nonwave1*U;
%
%     % get nonwave-specific subspace
%     [c,s,l]=pca(Lwave); % take the bottom 20PCs
%     Uwave_null = c(:,181:end);
%     Proj = Lnonwave*Uwave_null;
%     [c,s,l] = pca(Proj); % keep all the PCs at this point
%     Vnw_wavenull = c;
%     Z_nwunique = Uwave_null*Vnw_wavenull;
%
%     % get the percentage variance captured by nonwave specific subspace
%     Ywave  = Lwave*Z_nwunique;
%     Ynonwave  = Lnonwave*Z_nwunique;
%
%     % compute VAF
%     R=[];
%     for ii=1:size(Ywave,2)
%         tmp = Lwave - Lwave*Z_nwunique(:,ii)*Z_nwunique(:,ii)';
%         num=norm(Lwave,'fro')^2 - norm(tmp,'fro')^2;
%         den = norm(Lwave,'fro')^2;
%         R(ii) = num/den;
%     end
%     figure;stem(R*100)
% end

