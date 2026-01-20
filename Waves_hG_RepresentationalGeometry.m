%% LOOKING AT HOW REPRESENTATIONAL GEOMETRY OF HG CHANGES DURING
%  WAVE EPOCHS COMPARED TO NON WAVE EPOCHS

%% load subjects' data
clear
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
subj ='B3';
%parpool('threads')

if strcmp(subj,'B1')
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
    cd(root_path)
    load('ECOG_Grid_8596_000067_B3.mat')
    load B1_waves_stability_hG
    num_targets=7;

elseif strcmp(subj,'B3')
    num_targets=12;
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
    cd(root_path)
    load session_data_B3_Hand
    %load session_data_B3
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
    load B3_waves_hand_stability_Muller_hG


elseif strcmp(subj,'B6')
    num_targets=7;
    root_path='/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
    cd(root_path)
    %load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
    load('B6_waves_stability_Muller_hG')
end

%% ANALYSIS 0 
% Look at the mahalanobis distance between movements during wave epochs,
% non wave epochs, OL and CL 


%% ANALYSIS 0 and 1
% hg decouples of mu during waves as compared to non-waves. does the
% dynamic range of hg increase during wave epochs as opposed to non wave
% epochs? is there greater  total variance? 
% At same time, what does it say about the dimensionality of the manifold
% during wave vs. non wave epochs?
% manifold dimensionality: wave epochs have consistently lower
% dimensionality

res=[];res_var=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    dim_wave=[];
    dim_nonwave=[];
    for tid=1:num_targets
        act_wave=[];
        act_nonwave=[];
        for i=1:length(stats_cl_hg)
            if stats_cl_hg(i).target_id ==tid

                tmp = stats_cl_hg(i).hg_wave;
                tmp = cell2mat(tmp');
                act_wave = [act_wave;tmp];

                tmp = stats_cl_hg(i).hg_nonwave;
                tmp = cell2mat(tmp');
                act_nonwave = [act_nonwave;tmp];
            end
        end
        [c,s,l]=pca(zscore(act_wave));
        % dimensionality
        %pr_wave = ((sum(l))^2) / (sum(l.^2));
        vaf = cumsum(l)./sum(l);
        [aa bb]=find(vaf>0.8);
        pr_wave = aa(1); % z score data matrix

        % total variance
        %pr_wave = sum(log(l)); % dont z-score data matrix

        % dimensionality        
        idx=randperm(size(act_nonwave,1),size(act_wave,1));
        [c,s,l]=pca(zscore(act_nonwave(idx,:)));
        %pr_nonwave = ((sum(l))^2) / (sum(l.^2));
        vaf = cumsum(l)./sum(l);
        [aa bb]=find(vaf>0.8);
        pr_nonwave = aa(1); % z score data matrix

        % total variance
        %pr_nonwave = sum(l); % dont z-score data matrix
        %pr_nonwave = sum(log(l)); % dont z-score data matrix


        dim_wave=[dim_wave pr_wave];
        dim_nonwave=[dim_nonwave pr_nonwave];
    end
    res=[res;[mean(dim_wave) mean(dim_nonwave)]];
end
res
% dimensionality
res(:,1)-res(:,2)
signrank(ans)

% total variance
res(:,1)-res(:,2)
signrank(ans)

%% ANALYSIS 2
% hypothesis is that since hg decouples from mu, it has more expressiveness
% do it on a channel by channel basis.
% compute channel's mean activity within wave and nonwave epochs in single
% trials. look at the variability across trials. 
res_days=[];pval=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    D_wave=[];D_nonwave=[];res=[];
    for i=1:length(stats_cl_hg)
        if stats_cl_hg(i).target_id <=1
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');
            tmp = median(tmp,1);
            D_wave = cat(1,D_wave,tmp);

            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');
            tmp = median(tmp,1); % or mean here 
            D_nonwave = cat(1,D_nonwave,tmp);
        end
    end
    res = log([(std(D_wave,1))' (std(D_nonwave,1))']);
    % figure;
    % boxplot(res)
    [p,h] = signrank(res(:,1),res(:,2));
    pval(days) = ((p<0.05) * (median(res(:,1)) - median(res(:,2))))
    % xticks(1:2)
    % xticklabels({'Wave epochs','Non wave epochs'})
    % ylabel('Variability in mean activity across conditions')
    res_days =[res_days ;mean(res,1)];
    %res_days(days,:,:) = res;
end
[p,h] = signrank(res_days(:,1),res_days(:,2))
mean(res_days)
figure;boxplot(res_days)

%% ANALYSIS 3
% EXAMINING Representational geometry

res=[];res_var=[];res_ang=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    D_wave=[];D_nonwave=[];
    var_wave=[];
    var_nonwave=[];
    ang_wave=[];
    ang_nonwave=[];
    for tid0=1:num_targets
        for tid=tid0+1:num_targets
            act1=[];act2=[];
            act1_nonwave=[];act2_nonwave=[];
            for i=1:length(stats_cl_hg)
                if stats_cl_hg(i).target_id ==tid0
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    act1 = [act1;tmp];

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act1_nonwave = [act1_nonwave;tmp];
                end
                if stats_cl_hg(i).target_id ==tid
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    act2 = [act2;tmp];

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act2_nonwave = [act2_nonwave;tmp];
                end
            end
            d1 = mahal2((act1),(act2),2);
            d2 = mahal2((act1_nonwave),(act2_nonwave),2);
            D_wave = [D_wave d1];
            D_nonwave = [D_nonwave d2];

            % rep geometry, projection on axis
            m1 = mean(act1) - mean(act2);
            m2 = mean(act1_nonwave) - mean(act2_nonwave);
            w1 = m1./norm(m1);
            w2 = m2./norm(m2);
            proj1a = act1 * w1';
            proj1b = act2 * w1';
            var_on_axis_wave = var([proj1a; proj1b]);
            proj2a = act1_nonwave * w2';
            proj2b = act2_nonwave * w2';
            var_on_axis_nonwave = var([proj2a; proj2b]);

            var_wave =[var_wave;var_on_axis_wave];
            var_nonwave =[var_nonwave;var_on_axis_nonwave];

            % rep geometry looking at angle between top PCs and
            % task-subspace : this should be larger during waves

            c1a = cov(act1) + 1e-6*eye(size(act1,2));
            c1b = cov(act2) + 1e-6*eye(size(act2,2));
            c1 = 0.5*(c1a+c1b);
            c2a = cov(act1_nonwave) + 1e-6*eye(size(act1_nonwave,2));
            c2b = cov(act2_nonwave) + 1e-6*eye(size(act2_nonwave,2));
            c2 = 0.5*(c2a+c2b);

            [v1,d1]=eig(c1);d1=diag(d1);
            [d1, idx] = sort(d1, 'descend');
            v1=v1(:,idx);
            [v2,d2]=eig(c2);d2=(diag(d2));
            [d2, idx] = sort(d2, 'descend');
            v2=v2(:,idx);
            % sum(d1)
            % sum(d2)
            % sum(log(d1))
            % sum(log(d2))

            %ang_wave=[ang_wave; subspacea(v1(:,1:15),w1')];
            %ang_nonwave=[ang_wave;subspacea(v2(:,1:15),w2')];
            ang_wave=[ang_wave; norm(w1*v1(:,1:10))];
            ang_nonwave=[ang_wave;norm(w2*v2(:,1:10))];
        end
    end
    res(days,:) = [median(D_wave) median(D_nonwave)];
    res_var(days,:) = [median(var_wave) median(var_nonwave)];
    res_ang(days,:) = [median(ang_wave) median(ang_nonwave)];
end


figure;
boxplot(res)
[p,h] = signrank(res(:,1),res(:,2))
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Pairwise Mahalanobis dist.')
title('B1 hG Decoding Information')
plot_beautify


figure;
boxplot(res_var)
[p,h] = signrank(res_var(:,1),res_var(:,2));
[h,p,tb,st]=ttest(res_var(:,1),res_var(:,2));p
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Task specific variance')
title('B1 hG Decoding Information')
plot_beautify
tmp=(res_var(:,1)-res_var(:,2))./res_var(:,1);
[p,h]=signrank(tmp)
figure;boxplot(tmp)
hline(0)

figure;
boxplot(res_ang)
[p,h] = signrank(res_ang(:,1),res_ang(:,2));
[h,p,tb,st]=ttest(res_ang(:,1),res_ang(:,2));p
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Alignment index')
title(subj)
plot_beautify
tmp=(res_ang(:,1)-res_ang(:,2))./res_ang(:,1);
[p,h]=signrank(tmp)
figure;boxplot(tmp)
hline(0)



%%
%%% ANALYSIS 2
%%% VARIANCE IS HIGHER DURING CL -> LOOK AT ITS FUNCTION
% look at the angle between the task subspace (1d line connecting the
% means) and the subspace of highest VAF and subspace of lowest VAF.
% Same time also look at the variance of the data projected onto the
% task subspace

%%% ANALYSIS 2
% dimensionality of the manifold to capture 75% VAF, OL and CL, waves and
% non-waves, within each condition: trial to trial variance within
% condition



res_days=[];pval=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    D_wave=[];D_nonwave=[];res=[];
    for i=1:length(stats_cl_hg)
        if stats_cl_hg(i).target_id <=7
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');
            tmp = var(tmp,1);
            D_wave = cat(1,D_wave,tmp);

            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');
            tmp = var(tmp,1); % or mean here
            D_nonwave = cat(1,D_nonwave,tmp);
        end
    end
    res = log([(var(D_wave,1))' (var(D_nonwave,1))']);
    % figure;
    % boxplot(res)
    [p,h] = signrank(res(:,1),res(:,2));
    pval(days) = ((p<0.05) * (median(res(:,1)) - median(res(:,2))))
    % xticks(1:2)
    % xticklabels({'Wave epochs','Non wave epochs'})
    % ylabel('Variability in mean activity across conditions')
    res_days =[res_days ;mean(res,1)];
end
[p,h] = signrank(res_days(:,1),res_days(:,2))
mean(res_days)

%%%%% (MAIN) looking at hg differences between two conditions, wave vs. non wave


figure;
boxplot(res)
[p,h] = signrank(res(:,1),res(:,2))
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Pairwise Mahalanobis dist.')
title('B1 hG Decoding Information')
plot_beautify

figure;
boxplot(res_var)
[p,h] = signrank(res_var(:,1),res_var(:,2));
[h,p,tb,st]=ttest(res_var(:,1),res_var(:,2));
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Task specific variance')
title('B1 hG Decoding Information')
plot_beautify
tmp=(res_var(:,1)-res_var(:,2))./res_var(:,1);
[p,h]=signrank(tmp);
figure;boxplot(tmp)
hline(0)



c1a = cov(act1) + 1e-6*eye(size(act1,2));
c1b = cov(act2) + 1e-6*eye(size(act2,2));
c1 = 0.5*(c1a+c1b);
c2a = cov(act1_nonwave) + 1e-6*eye(size(act1_nonwave,2));
c2b = cov(act2_nonwave) + 1e-6*eye(size(act2_nonwave,2));
c2 = 0.5*(c2a+c2b);

[v1,d1]=eig(c1);d1=diag(d1);
[v2,d2]=eig(c2);d2=diag(d2);
sum(d1)
sum(d2)
sum(log(d1))
sum(log(d2))

m1 = mean(act1) - mean(act2);
m2 = mean(act1_nonwave) - mean(act2_nonwave);
D1 = pdist([m1;v1(:,1)'],'cosine');
D2 = pdist([m2;v2(:,1)'],'cosine');
w1 = m1./norm(m1);
w2 = m2./norm(m2);
proj1a = act1 * w1';
proj1b = act2 * w1';
var_on_axis_wave = var([proj1a; proj1b]);
proj2a = act1_nonwave * w2';
proj2b = act2_nonwave * w2';
var_on_axis_nonwave = var([proj2a; proj2b]);


