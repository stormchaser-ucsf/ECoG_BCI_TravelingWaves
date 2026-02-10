%% LOOKING AT HOW REPRESENTATIONAL GEOMETRY OF HG CHANGES DURING
%  WAVE EPOCHS COMPARED TO NON WAVE EPOCHS

%% WAVE PROCESSING SUBJECTS' DATA



%% load subjects' data
clear;
clc
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))
subj ='B3';
%parpool('threads')

if strcmp(subj,'B1')
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
    cd(root_path)
    load('ECOG_Grid_8596_000067_B3.mat')
    %load B1_waves_stability_hG
    %load B1_waves_stability_hG_plv
    %load B1_waves_stability_hgFilterBank_PLV_AccStatsCL
    load B1_waves_stability_hgFilterBank_PLV_AccStatsCL_v2
    num_targets=7;

elseif strcmp(subj,'B3')    
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
    cd(root_path)
    load session_data_B3_Hand
    %load session_data_B3
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
    %load B3_waves_hand_stability_Muller_hG
    %load B3_waves_hand_stability_Muller_hG_plv
    %load B3_waves_stability_hgFilterBank_PLV_AccStatsCL
    %load B3_waves_3DArrow_stability_hgFilterBank_PLV_AccStatsCL
    load B3_waves_3DArrow_stability_hgFilterBank_PLV_AccStatsCL_v2
    num_targets=7;


elseif strcmp(subj,'B6')
    num_targets=7;
    root_path='/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
    cd(root_path)
    %load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
    %load('B6_waves_stability_Muller_hG')
    %load B6_waves_stability_hg_PLV
    %load B6_waves_stability_hgFilterBank_PLV_AccStatsCL
    load B6_waves_stability_hgFilterBank_PLV_AccStatsCL_v2_AllData
    num_targets=7;
end

%% ANALYSIS -2 
%MAHAB Distances
% B3 -> flipud 

% looking at hg differences between two conditions, wave vs. non wave
res=[];
for days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    stats_cl = stats_cl_days{days};
    D_wave=[];D_nonwave=[];
    for tid0=1:7
        for tid=tid0+1:7
            act1=[];act2=[];
            act1_nonwave=[];act2_nonwave=[];
            for i=1:length(stats_cl_hg)
                if stats_cl_hg(i).target_id ==tid0 
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');

                    %if stats_cl(i).accuracy==0
                        act1 = [act1;tmp];
                    %end

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act1_nonwave = [act1_nonwave;tmp];
                end
                if stats_cl_hg(i).target_id ==tid 
                    tmp = stats_cl_hg(i).hg_wave;
                    tmp = cell2mat(tmp');
                    %if stats_cl(i).accuracy==0
                        act2 = [act2;tmp];
                    %end

                    tmp = stats_cl_hg(i).hg_nonwave;
                    tmp = cell2mat(tmp');
                    act2_nonwave = [act2_nonwave;tmp];
                end
            end
            d1 = mahal2(act1,act2,2);
            d2 = mahal2(act1_nonwave,act2_nonwave,2);
            D_wave = [D_wave d1];
            D_nonwave = [D_nonwave d2];
        end
    end
    res(days,:) = [median(D_wave) median(D_nonwave)];
end
figure;boxplot(res)
[p,h] = signrank(res(:,1),res(:,2))
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Pairwise Mahalanobis dist.')
title(subj)
plot_beautify

figure;plot(res)

%% ANALYSIS -1 
% PLV between mu and hg


%%%%%% JUST FOR CLOSED LOOP, TRIAL LENGTH MATCHING
elec_list=[14	8	2	130	136	142	147	151	18	13	7	1	129	135	141	146
    10	4	132	138	144	149	153	156	158	27	24	20	15	9	3	131
    189	192	32	31	29	26	22	17	11	5	133	139	145	150	154	157
    169	173	177	181	185	188	191	64	61	58	54	50	46	42	12	6
    40	37	34	162	165	168	172	176	180	184	187	190	63	60	57	53
    89	55	51	47	43	39	36	33	161	164	167	171	175	179	183	217
    212	215	219	223	94	90	86	83	80	77	73	69	65	193	197	201
    195	199	203	207	210	213	216	220	224	95	91	87	84	81	78	74
    114	109	76	72	68	196	200	204	237	242	247	251	254	256	96	92
    244	249	253	127	124	120	115	110	105	101	97	225	229	233	238	243
    122	117	112	107	103	99	227	231	235	240	245	250	125	121	116	111];

for i=1:numel(elec_list)
    %disp([i elec_list(i)])
    if elec_list(i)>=109 && elec_list(i)<=112
        elec_list(i) = elec_list(i)-1;
    elseif elec_list(i)>=114 && elec_list(i)<=117
        elec_list(i) = elec_list(i)-2;
    elseif elec_list(i)>=119
        elec_list(i) = elec_list(i)-3;
    end
end

res_days=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_ol_hg_days{days};
    wave_len_cl=[];
    nonwave_len_cl=[];
    wave_plv=[];
    nonwave_plv=[];
    for i=1:length(stats_cl_hg)
        %%% just straight up average plv across grid
        a=stats_cl_hg(i).plv_wave;
        %a=a(:,elec_list);
        wave_len_cl(i) = size(a,1);

        b = stats_cl_hg(i).plv_nonwave;
        %b=b(:,elec_list);
        nonwave_len_cl(i) = size(b,1);

        % nonwave_plv(i) = mean(abs(mean(b)));
        % wave_plv(i) = mean(abs(mean(a)));

        if wave_len_cl(i)<nonwave_len_cl(i)
            wave_plv(i) = mean(abs(mean(a)));

            len = min(30,nonwave_len_cl(i) -  wave_len_cl(i));
            idx=randperm(nonwave_len_cl(i) -  wave_len_cl(i),len);
            plv_tmp=[];
            for j=1:length(idx)
                tmp = b(idx(j):idx(j)+wave_len_cl(i)-1,:);
                plv_tmp(j) = mean(abs(mean(tmp,1)));
            end
            nonwave_plv(i) = mean(plv_tmp);

        elseif wave_len_cl(i)>nonwave_len_cl(i)
            nonwave_plv(i) = mean(abs(mean(b)));

            len = min(30,wave_len_cl(i) -  nonwave_len_cl(i));
            idx=randperm(wave_len_cl(i) -  nonwave_len_cl(i),len);
            plv_tmp=[];
            for j=1:length(idx)
                tmp = a(idx(j):idx(j)+nonwave_len_cl(i)-1,:);
                plv_tmp(j) = mean(abs(mean(tmp,1)));
            end
            wave_plv(i) = mean(plv_tmp);

        elseif wave_len_cl(i)== nonwave_len_cl(i)
            nonwave_plv(i) = mean(abs(mean(b)));
            wave_plv(i) = mean(abs(mean(a)));
        end
    end
    res_days(days,:) = [mean(wave_plv) mean(nonwave_plv)];
end
% figure;
% boxplot([wave_plv' nonwave_plv'])
% xticks(1:2)
% xticklabels({'Wave epochs','Non wave epochs'})
% plot_beautify
% [p,h] = signrank(wave_plv,nonwave_plv);
% title(num2str(p))

figure;
boxplot(res_days)
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
plot_beautify
[p,h] = signrank(res_days(:,1),res_days(:,2));
title(num2str(p))

figure;plot(res_days)
legend('Waves','Nonwaves')

%% ANALYSIS 0 and 1 -> VARIANCE, DIMENSIONAL COLLAPSE PCA style (MAIN)
% hg decouples of mu during waves as compared to non-waves. does the
% dynamic range of hg increase during wave epochs as opposed to non wave
% epochs? is there greater  total variance? 
% At same time, what does it say about the dimensionality of the manifold
% during wave vs. non wave epochs?
% manifold dimensionality: wave epochs have consistently lower
% dimensionality -> there is dimensional collapse


cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
load('B1_waves_stability_hgFilterBank_PLV_AccStatsCL_v2.mat','stats_cl_hg_days')
[res_days_B1] = get_PCs_VAF_DimCollapse(stats_cl_hg_days,7)

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
load('B3_waves_3DArrow_stability_hgFilterBank_PLV_AccStatsCL_v2.mat','stats_cl_hg_days')
[res_days_B3] = get_PCs_VAF_DimCollapse(stats_cl_hg_days,7)

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
load('B6_waves_stability_hgFilterBank_PLV_AccStatsCL_v2_AllData.mat','stats_cl_hg_days')
[res_days_B6] = get_PCs_VAF_DimCollapse(stats_cl_hg_days(1:end-1),7)


% plotting

%figure;boxplot(res)


res_days = [res_days_B1;res_days_B3;res_days_B6];
figure;hold on
%b1
idx = [ones(size(res_days_B1,1),1) 2*ones(size(res_days_B1,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B1(:,1),50,'b','LineWidth',1)
scatter(idx(:,2),res_days_B1(:,2),50,'b',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B1(i,1),res_days_B1(i,2)],...
        'Color',[0 0 1 .25])
end

%b3
%figure;hold on
idx = [ones(size(res_days_B3,1),1) 2*ones(size(res_days_B3,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B3(:,1),50,'r','LineWidth',1)
scatter(idx(:,2),res_days_B3(:,2),50,'r',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B3(i,1),res_days_B3(i,2)],...
        'Color',[1 0 0 .25])
end

%b6
%figure;hold on
idx = [ones(size(res_days_B6,1),1) 2*ones(size(res_days_B6,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B6(:,1),50,'k','LineWidth',1)
scatter(idx(:,2),res_days_B6(:,2),50,'k',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B6(i,1),res_days_B6(i,2)],...
        'Color',[0.25 0.25 0.25 .25])
end
xlim([.5 2.5])
boxplot(res_days)
xticks(1:2)
xticklabels({'Wave epoch hG','Nonwave epoch hG'})
ylabel('Number of PCs for 80% VAF')
plot_beautify
%ylim([25 70])

signrank(res_days(:,1),res_days(:,2))

% figure;hold on
% stem(cumsum(l0)./sum(l0),'LineWidth',1)
% stem(cumsum(l)./sum(l),'LineWidth',1)
% legend({'Wave epochs','Nonwave epochs'})
% plot_beautify
% ylabel('VAF')
% xlabel('PC Number')
% hline(0.8,'--k')

%% ANALYSIS 2 : VARIANCE CHANNEL STYLE (MAIN)
% hypothesis is that since hg decouples from mu, it has more expressiveness
% do it on a channel by channel basis.
% compute channel's mean activity within wave and nonwave epochs in single
% trials. look at the variability across trials. 

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
load('B1_waves_stability_hgFilterBank_PLV_AccStatsCL_v2.mat',...
    'stats_cl_hg_days','subj')
[res_days_B1] = get_Channel_TrialVariance(stats_cl_hg_days,7)

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
load('B3_waves_3DArrow_stability_hgFilterBank_PLV_AccStatsCL_v2.mat',...
    'stats_cl_hg_days','subj')
[res_days_B3] = get_Channel_TrialVariance(stats_cl_hg_days,7)

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
load('B6_waves_stability_hgFilterBank_PLV_AccStatsCL_v2_AllData.mat',...
    'stats_cl_hg_days','subj')
[res_days_B6] = get_Channel_TrialVariance(stats_cl_hg_days(1:end-1),7)




res_days = [res_days_B1;res_days_B3;res_days_B6];
figure;hold on
%b1
idx = [ones(size(res_days_B1,1),1) 2*ones(size(res_days_B1,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B1(:,1),50,'b','LineWidth',1)
scatter(idx(:,2),res_days_B1(:,2),50,'b',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B1(i,1),res_days_B1(i,2)],...
        'Color',[0 0 1 .25])
end

%b3
figure;hold on
idx = [ones(size(res_days_B3,1),1) 2*ones(size(res_days_B3,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B3(:,1),50,'r','LineWidth',1)
scatter(idx(:,2),res_days_B3(:,2),50,'r',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B3(i,1),res_days_B3(i,2)],...
        'Color',[1 0 0 .25])
end

%b6
figure;hold on
idx = [ones(size(res_days_B6,1),1) 2*ones(size(res_days_B6,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B6(:,1),50,'k','LineWidth',1)
scatter(idx(:,2),res_days_B6(:,2),50,'k',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B6(i,1),res_days_B6(i,2)],...
        'Color',[0.25 0.25 0.25 .25])
end
xlim([.5 2.5])

boxplot(res_days)
xticks(1:2)
xticklabels({'Wave epoch hG','Nonwave epoch hG'})
ylabel('Trial to Trial variance in mean')
plot_beautify
%ylim([25 70])

signrank(res_days(:,1),res_days(:,2))


res_days=[];pval=[];
res_std_days=[];
for days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    D_wave=[];D_nonwave=[];res=[];
    D_wave_total=[];D_nonwave_total=[];res_std=[];
    for i=1:length(stats_cl_hg)
        if stats_cl_hg(i).target_id <=num_targets
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');tmp1=tmp;
            tmp = mean(tmp,1);
            if ~isempty(tmp) && sum(~isnan(tmp)) == 253
                D_wave = cat(1,D_wave,tmp);
                D_wave_total = [D_wave_total;tmp1];
            end

            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');tmp1=tmp;
            tmp = mean(tmp,1); % or mean here or var
            if ~isempty(tmp) && sum(~isnan(tmp)) == 253
                D_nonwave = cat(1,D_nonwave,tmp);
                D_nonwave_total = [D_nonwave_total;tmp1];
            end
        end
    end
    %res = ([(var(D_wave,1))' (var(D_nonwave,1))']);
    res = log([(std(D_wave,1))' (std(D_nonwave,1))']);
    res_std = [(log(std(D_wave_total,1)))' (log(std(D_nonwave_total,1)))'];
    % figure;
    % boxplot(res)
    [p,h] = signrank(res(:,1),res(:,2));
    pval(days) = ((p<0.05) * (median(res(:,1)) - median(res(:,2))));
    % xticks(1:2)
    % xticklabels({'Wave epochs','Non wave epochs'})
    % ylabel('Variability in mean activity across conditions')
    res_days =[res_days ;mean(res,1)];
    res_std_days=[res_std_days;median(res_std,1)];
    %res_days(days,:,:) = res;
end
[p,h] = signrank(res_days(:,1),res_days(:,2))
mean(res_days)
%figure;boxplot(res_days,'Notch','off')
idx=[0.01*randn(size(res_days,1),1) + ones(size(res_days,1),1)...
    0.01*randn(size(res_days,1),1) + 2*ones(size(res_days,1),1)];
figure;scatter(idx,res_days)
xlim([0.5 2.5])
hold on
for i=1:size(res_days,1)
    plot([idx(i,1) idx(i,2)],[res_days(i,1) res_days(i,2)],'Color',[.5 .5 .5 .5]);
end
res_days_cl=res_days;
xticks(1:2)
xticklabels({'Wave','Non wave'})
ylabel('Trial to Trial Variance')

res_std_days_cl=res_std_days;


res_days=[];pval=[];
res_std_days=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_ol_hg_days{days};
    D_wave=[];D_nonwave=[];res=[];
    D_wave_total=[];D_nonwave_total=[];res_std=[];
    for i=1:length(stats_cl_hg)
        if stats_cl_hg(i).target_id <=7
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');tmp1=tmp;
            tmp = mean(tmp,1);
            if ~isempty(tmp) && sum(~isnan(tmp)) == 253
                D_wave = cat(1,D_wave,tmp);
                D_wave_total = [D_wave_total;tmp1];
            end

            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');tmp1=tmp;
            tmp = mean(tmp,1); % or mean, var here
            if ~isempty(tmp) && sum(~isnan(tmp)) == 253
                D_nonwave = cat(1,D_nonwave,tmp);
                D_nonwave_total = [D_nonwave_total;tmp1];
            end
        end
    end
    %res = ([(var(D_wave,1))' (var(D_nonwave,1))']);
    res = log([(std(D_wave,1))' (std(D_nonwave,1))']);
    res_std = [(log(std(D_wave_total,1)))' (log(std(D_nonwave_total,1)))'];
    % figure;
    % boxplot(res)
    [p,h] = signrank(res(:,1),res(:,2));
    pval(days) = ((p<0.05) * (median(res(:,1)) - median(res(:,2))))
    % xticks(1:2)
    % xticklabels({'Wave epochs','Non wave epochs'})
    % ylabel('Variability in mean activity across conditions')
    res_days =[res_days ;mean(res,1)];
    res_std_days=[res_std_days;median(res_std,1)];
    %res_days(days,:,:) = res;
end
[p,h] = signrank(res_days(:,1),res_days(:,2))
mean(res_days)
%figure;boxplot(res_days)
res_days_ol=res_days;
res_std_days_ol=res_std_days;

res=[res_days_ol res_days_cl];
figure;boxplot(res)
xticks(1:4)
xticklabels({'OL wave epochs ','OL nonwave epochs','CL wave epochs',...
    'CL nonwave epochs'})
ylabel('Grid-wise across trial variance in mean hG (log)')
plot_beautify
title(subj)


res=[res_std_days_ol res_std_days_cl];
figure;boxplot(res)
xticks(1:4)
xticklabels({'OL wave epochs ','OL nonwave epochs','CL wave epochs',...
    'CL nonwave epochs'})
ylabel('Grid-wise variance (log)')
plot_beautify
title(subj)
signrank([res(:,1);res(:,2)],[res(:,3);res(:,4)])

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
            % for i=1:size(act1,1)
            %     act1(i,:)=act1(i,:)./norm(act1(i,:));                
            % end
            % for i=1:size(act2,1)
            %     act2(i,:)=act2(i,:)./norm(act2(i,:));                
            % end
            % for i=1:size(act1_nonwave,1)
            %     act1_nonwave(i,:)=act1_nonwave(i,:)./norm(act1_nonwave(i,:));                
            % end
            % for i=1:size(act2_nonwave,1)
            %     act2_nonwave(i,:)=act2_nonwave(i,:)./norm(act2_nonwave(i,:));                
            % end

            %if size(act1_nonwave,1) > size(act1,1)
            act1_nonwave=act1_nonwave(1:size(act1,1),:);
            %elseif size(act1_nonwave,1) < size(act1,1)
            %    act1=act1(1:size(act1_nonwave,1),:);
            %end

            %if size(act2_nonwave,1) > size(act2,1)
            act2_nonwave=act2_nonwave(1:size(act2,1),:);
            %elseif size(act2_nonwave,1) < size(act2,1)
            %    act2=act2(1:size(act2_nonwave,1),:);
            %end

            m  = mean([act1;act2;act1_nonwave;act2_nonwave]);
            s = std([act1;act2;act1_nonwave;act2_nonwave]);
            act1 = (act1-m)./s;
            act2 = (act2-m)./s;
             
            act1_nonwave = (act1_nonwave-m)./s;
            act2_nonwave = (act2_nonwave-m)./s;

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
            ang_wave=[ang_wave; norm(w1*v1(:,1:50))];
            ang_nonwave=[ang_wave;norm(w2*v2(:,1:50))];
        end
    end
    res(days,:) = [mean(D_wave) mean(D_nonwave)];
    res_var(days,:) = [mean(var_wave) mean(var_nonwave)];
    res_ang(days,:) = [mean(ang_wave) mean(ang_nonwave)];
end


figure;
boxplot((res))
%res=log(res)
[p,h] = signrank(res(:,1),res(:,2))
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Pairwise Mahalanobis dist.')
title(subj)
plot_beautify


figure;
boxplot(res_var)
[p,h] = signrank(res_var(:,1),res_var(:,2));
[h,p,tb,st]=ttest(res_var(:,1),res_var(:,2));p
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Task specific variance')
title(subj)
plot_beautify
tmp=(res_var(:,1)-res_var(:,2))./res_var(:,1);
[p,h]=signrank(tmp)
figure;boxplot(tmp)
hline(0)
plot_beautify
title(subj)
ylabel('Task specific variance')

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
title(subj)
plot_beautify
ylabel('Alignment index')

%% ANALYSIS 3   
% continued from above
% using LDA to see classification accuracy between two movements, trial
% level

%stats_cl=stats_cl_days{8};
%stats_cl_hg = stats_cl_hg_days{8};

res=[];
for iter=1:25
    A={};
    B={};
    for i=1:length(stats_cl)
        if stats_cl(i).target_id == 7
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');
            A = cat(1,A,tmp);
        end

        if stats_cl(i).target_id == 6
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');
            B = cat(1,B,tmp);
        end
    end
    % 2 vs. 6 there are 2038 wave, 3016 nonwave


    total_trials = [A;B];
    %total_idx=[zeros(length(A),1);ones(length(B))];
    if length(A)<length(B)
        aa=randperm(length(B),length(A));
        B=B(aa);
    elseif length(A)>length(B)
        aa=randperm(length(A),length(B));
        A=A(aa);
    end

    train_idx = randperm(length(A),round(0.8*(length(A))));
    I = ones(length(A),1);
    I(train_idx)=0;
    test_idx = find(I==1);

    trainA = A(train_idx);
    trainA=cell2mat(trainA);
    lend = 6000-size(trainA,1);
    xx = randi(size(trainA,1),lend,1);
    tmpA = trainA(xx,:);
    tmpA = 0.01*randn(size(tmpA)) + tmpA;
    trainA = [trainA;tmpA];
    idxA = zeros(size(trainA,1),1);

    trainB = B(train_idx);
    trainB=cell2mat(trainB);
    lend = 5000-size(trainB,1);
    xx = randi(size(trainB,1),lend,1);
    tmpB = trainB(xx,:);
    tmpB = 0.01*randn(size(tmpB)) + tmpB;
    trainB = [trainB;tmpB];
    idxB = ones(size(trainB,1),1);

    if size(trainA,1)>size(trainB,1)
        idx = randperm(size(trainA,1),size(trainB,1));
        trainA = trainA(idx,:);
        idxA = idxA(idx,:);
    elseif size(trainA,1)<size(trainB,1)
        idx = randperm(size(trainB,1),size(trainA,1));
        trainB = trainB(idx,:);
        idxB = idxB(idx,:);
    end

    X = [trainA;trainB];
    Y = [idxA;idxB];

    %3016 data points. Have to bring it down to 2038 (1019 per class).
    % ca = randperm(length(idxA),1300);
    % cb = randperm(length(idxB),1300);
    % trainA = trainA(ca,:);
    % trainB = trainB(ca,:);
    % idxA = idxA(cb,:);
    % idxB = idxB(cb,:);
    % X = [trainA;trainB];
    % Y = [idxA;idxB];

    % aa = randperm(size(X,1),2656);
    % X = X(aa,:);
    % Y = Y(aa,:);

    W = LDA(X,Y);

    % Mdl = fitcdiscr(X, Y, ...
    %     'DiscrimType','linear');

    % test on held out samples
    testA = A(test_idx);
    testA=cell2mat(testA);
    idxA = zeros(size(testA,1),1);
    testB = B(test_idx);
    testB=cell2mat(testB);
    idxB = ones(size(testB,1),1);

    Xtest = [testA;testB];
    Ytest = [idxA;idxB];
    len = length(Ytest);

    L = [ones(len,1) Xtest] * W';
    P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);   
    decodes=[];
    for i=1:size(P,1)
        tmp = P(i,:);
        [aa bb]=max(tmp);
        decodes(i) = bb-1;
    end

    % decodes = predict(Mdl,Xtest);
    % acc = mean(decodes==Ytest);


    res(iter) = sum(Ytest==decodes')/length(decodes);
    %res(iter)=calculateBalancedAccuracy(Ytest,decodes');
end

mean(res)

% 
% k=51;
% figure;plot(zscore(stats_cl(k).stab))
% hline(0)
% hold on
% plot((stats_cl(k).output))
% [out,st,stp]=wave_stability_detect(zscore(stats_cl(k).stab))
% vline(st,'g')
% vline(stp,'r')
%% ANALYSIS 3.5
% LDA across all days rather than just within day

% load the data across all days
A={};B={};
for days=1:length(stats_cl_hg_days)
    stats_cl = stats_cl_days{days};
    stats_cl_hg = stats_cl_hg_days{days};
    for i=1:length(stats_cl)
        if stats_cl(i).target_id == 1
            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');
            A = cat(1,A,tmp);
        end

        if stats_cl(i).target_id == 6
            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');
            B = cat(1,B,tmp);
        end
    end    
end

res=[];
for iter=1:25
    total_trials = [A;B];
    %total_idx=[zeros(length(A),1);ones(length(B))];
    if length(A)<length(B)
        aa=randperm(length(B),length(A));
        B=B(aa);
    elseif length(A)>length(B)
        aa=randperm(length(A),length(B));
        A=A(aa);
    end

    train_idx = randperm(length(A),round(0.8*(length(A))));
    I = ones(length(A),1);
    I(train_idx)=0;
    test_idx = find(I==1);

    trainA = A(train_idx);
    trainA=cell2mat(trainA);
    % lend = 6000-size(trainA,1);
    % xx = randi(size(trainA,1),lend,1);
    % tmpA = trainA(xx,:);
    % tmpA = 0.01*randn(size(tmpA)) + tmpA;
    % trainA = [trainA;tmpA];
    idxA = zeros(size(trainA,1),1);

    xx=zscore(var(trainA,1));
    [aa bb]=find(abs(xx)>3);
    trainA(:,bb) = 0.001*randn(size(trainA,1),length(bb));

    trainB = B(train_idx);
    trainB=cell2mat(trainB);
    % lend = 5000-size(trainB,1);
    % xx = randi(size(trainB,1),lend,1);
    % tmpB = trainB(xx,:);
    % tmpB = 0.01*randn(size(tmpB)) + tmpB;
    % trainB = [trainB;tmpB];
    idxB = ones(size(trainB,1),1);

    xx=zscore(var(trainB,1));
    [aa bb]=find(abs(xx)>3);
    trainB(:,bb) = 0.001*randn(size(trainB,1),length(bb));

    % if size(trainA,1)>size(trainB,1)
    %     idx = randperm(size(trainA,1),size(trainB,1));
    %     trainA = trainA(idx,:);
    %     idxA = idxA(idx,:);
    % elseif size(trainA,1)<size(trainB,1)
    %     idx = randperm(size(trainB,1),size(trainA,1));
    %     trainB = trainB(idx,:);
    %     idxB = idxB(idx,:);
    % end

    X = [trainA;trainB];
    Y = [idxA;idxB];

    %3016 data points. Have to bring it down to 2038 (1019 per class).
    % ca = randperm(length(idxA),1300);
    % cb = randperm(length(idxB),1300);
    % trainA = trainA(ca,:);
    % trainB = trainB(ca,:);
    % idxA = idxA(cb,:);
    % idxB = idxB(cb,:);
    % X = [trainA;trainB];
    % Y = [idxA;idxB];

    % aa = randperm(size(X,1),2656);
    % X = X(aa,:);
    % Y = Y(aa,:);

    W = LDA(X,Y);

    % Mdl = fitcdiscr(X, Y, ...
    %     'DiscrimType','linear');

    % test on held out samples
    testA = A(test_idx);
    testA=cell2mat(testA);
    idxA = zeros(size(testA,1),1);
    testB = B(test_idx);
    testB=cell2mat(testB);
    idxB = ones(size(testB,1),1);

    xx=zscore(var(testA,1));
    [aa bb]=find(abs(xx)>3);
    testA(:,bb) = 0.001*randn(size(testA,1),length(bb));

    xx=zscore(var(testB,1));
    [aa bb]=find(abs(xx)>3);
    testB(:,bb) = 0.001*randn(size(testB,1),length(bb));




    Xtest = [testA;testB];
    Ytest = [idxA;idxB];
    len = length(Ytest);

    L = [ones(len,1) Xtest] * W';
    P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);
    decodes=[];
    for i=1:size(P,1)
        tmp = P(i,:);
        [aa bb]=max(tmp);
        decodes(i) = bb-1;
    end

    % decodes = predict(Mdl,Xtest);
    % acc = mean(decodes==Ytest);


    %res(iter) = sum(Ytest==decodes')/length(decodes);
    res(iter)=calculateBalancedAccuracy(Ytest,decodes');

end
mean(res)


%% ANALYSIS 3.75
% participation ratio of e.values between wave and non wave epochs data,
% condition by condition

res=[];
for condn=1:num_targets
    A={};B={};
    for days=1:length(stats_cl_hg_days)
        stats_cl = stats_cl_days{days};
        stats_cl_hg = stats_cl_hg_days{days};
        for i=1:length(stats_cl)
            if stats_cl(i).target_id == condn
                tmp = stats_cl_hg(i).hg_nonwave;
                tmp = cell2mat(tmp');
                A = cat(1,A,tmp);
            end

            if stats_cl(i).target_id == condn
                tmp = stats_cl_hg(i).hg_wave;
                tmp = cell2mat(tmp');
                B = cat(1,B,tmp);
            end
        end
    end
    A=cell2mat(A);
    B=cell2mat(B);

    xx=zscore(var(A,1));
    [aa bb]=find(abs(xx)>3);
    A(:,bb) = 0.001*randn(size(A,1),length(bb));

    xx=zscore(var(B,1));
    [aa bb]=find(abs(xx)>3);
    B(:,bb) = 0.001*randn(size(B,1),length(bb));

    if size(A,1) > size(B,1)
        idx = randperm(size(A,1),size(B,1));
        A = A(idx,:);
        %A=A(end-size(B,1)+1:end,:);
    elseif size(A,1) < size(B,1)
        idx = randperm(size(B,1),size(A,1));
        B = B(idx,:);
    end

    [c,s,l]=pca(A);
    prA = sum(l)^2/sum(l.^2);

    [c1,s1,l1]=pca(B);
    prB = sum(l1)^2/sum(l1.^2);

    res = [res;[prA prB]];
end

figure;boxplot(res)
xticks(1:2)
xticklabels({'Non Wave','Wave'})
signrank(res(:,1),res(:,2))
mean(res)
plot_beautify
title(subj)
ylabel('PR')

%% ANALYSIS 4 (MAIN MAIN)
% build a classifer across days looking at decoding performance in hg
% during wave vs. non wave epochs

% B3,B6 -> no smoothing
% B1 -> smoothing 

addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))

num_targets=7;

if strcmp(subj,'B6')
    stats_cl_days = stats_cl_days(1:end-1);
    stats_cl_hg_days = stats_cl_hg_days(1:end-1);
end

% wave
% have to get the data into a cell array of condn_data, with target id and
% neural features
condn_data={};k=1;
for days = 1:length(stats_cl_days)
    stats_cl = stats_cl_days{days};
    stats_cl_hg = stats_cl_hg_days{days};
    for i=1:length(stats_cl)
        tid=stats_cl(i).target_id;
        if tid<=num_targets
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');

            % for j=1:size(tmp,2)
            %     tmp(:,j) =smooth(tmp(:,j),10);
            % end

            % normalize trial length to that of non waves 
            % tmp1 = stats_cl_hg(i).hg_nonwave;
            % tmp1 = cell2mat(tmp1');

            idx = randperm(size(tmp,1),20);
            tmp = tmp(idx,:);

            

            % if size(tmp,1) > size(tmp1,1)
            %     idx = randperm(size(tmp,1),size(tmp1,1));
            %     %idx = 1:size(tmp1,1);
            %     tmp = tmp(idx,:);
            % end


            % store wave stability values
            % stab = stats_cl(i).stab;
            % [out,st,stp] = wave_stability_detect(zscore(stab));
            % wave_stab=[];
            % for j=1:length(st)
            %     wave_stab=[wave_stab;stab(st(k):stp(k))];
            % end
            condn_data(k).neural = tmp';
            condn_data(k).targetID = tid;
            k=k+1;
        end
    end
end

condn_data1={};k=1;
for i=1:length(condn_data)
    if ~isempty(condn_data(i).neural)
        condn_data1(k).neural = condn_data(i).neural;
        condn_data1(k).targetID = condn_data(i).targetID;
        k=k+1;
    end
end
condn_data=condn_data1;

iterations=10;
[acc_wave,train_permutations,acc_bin_wave,bino_pdf,bino_pdf_chance] = ...
    accuracy_imagined_data(condn_data, iterations);
%accuracy_imagined_data_Hand_B3
%accuracy_imagined_data_Hand_B3

% non wave
condn_data={};k=1;
for days = 1:length(stats_cl_days)
    stats_cl = stats_cl_days{days};
    stats_cl_hg = stats_cl_hg_days{days};
    for i=1:length(stats_cl)
        tid=stats_cl(i).target_id;
        if tid<=num_targets
            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');

            % for j=1:size(tmp,2)
            %     tmp(:,j) =smooth(tmp(:,j),10);
            % end
            % 
            % if size(tmp,1)>62
            %     tmp = tmp(1:62,:);
            % end

            % normalize trial length to that of non waves 
            % tmp1 = stats_cl_hg(i).hg_wave;
            % tmp1 = cell2mat(tmp1');

            idx = randperm(size(tmp,1),20);
            tmp = tmp(idx,:);

            % if size(tmp,1) > size(tmp1,1)
            %     idx = randperm(size(tmp,1),size(tmp1,1));
            %     %idx = 1:size(tmp1,1);
            %     tmp = tmp(idx,:);
            % end



            condn_data(k).neural = tmp';
            condn_data(k).targetID = tid;
            k=k+1;
        end
    end
end


condn_data1={};k=1;
for i=1:length(condn_data)
    if ~isempty(condn_data(i).neural)
        condn_data1(k).neural = condn_data(i).neural;
        condn_data1(k).targetID = condn_data(i).targetID;
        k=k+1;
    end
end
condn_data=condn_data1;

[acc_nonwave,train_permutations,acc_bin_nonwave,bino_pdf,bino_pdf_chance] = ...
    accuracy_imagined_data(condn_data, iterations);

% % non wave, least global wave patterns
% condn_data={};k=1;
% for days = 1:length(stats_cl_days)
%     stats_cl = stats_cl_days{days};
%     stats_cl_hg = stats_cl_hg_days{days};
%     for i=1:length(stats_cl)
%         tid=stats_cl(i).target_id;
%         if tid<=num_targets
%             tmp = stats_cl_hg(i).hg_nonwave;
%             stab = stats_cl_hg(i).nonwave_stab;
%             stab_val=[];
%             for j=1:length(stab)
%                 stab_val(j) = median(cell2mat(stab(j)));
%             end
%             [aa bb] = sort(stab_val);
%             len = min(3,length(bb));
%             bb = sort(bb(1:len)); 
% 
%             tmp = cell2mat(tmp(bb)');
% 
%             for j=1:size(tmp,2)
%                 tmp(:,j) =smooth(tmp(:,j),10);
%             end
%             % 
%             condn_data(k).neural = tmp';
%             condn_data(k).targetID = tid;
%             k=k+1;
%         end
%     end
% end
% 
% 
% condn_data1={};k=1;
% for i=1:length(condn_data)
%     if ~isempty(condn_data(i).neural)
%         condn_data1(k).neural = condn_data(i).neural;
%         condn_data1(k).targetID = condn_data(i).targetID;
%         k=k+1;
%     end
% end
% condn_data=condn_data1;
% 
% [acc_nonwave_least,train_permutations,acc_bin_nonwave_least,...
%     bino_pdf,bino_pdf_chance] = ...
%     accuracy_imagined_data(condn_data, iterations);


%accuracy_imagined_data
%accuracy_imagined_data_Hand_B3

res=[];
for i=1:size(acc_nonwave,1)
    %tmp0=squeeze((acc_nonwave_least(i,:,:)));
    tmp=squeeze((acc_nonwave(i,:,:)));
    tmp1=squeeze((acc_wave(i,:,:)));
    %res(i,:) = [mean(diag(tmp0)) mean(diag(tmp)) mean(diag(tmp1))];
    res(i,:) = [ mean(diag(tmp)) mean(diag(tmp1))];
end
figure;boxplot(res)
xticks(1:2)
xticklabels({'Non wave epochs', 'Wave epochs'})
%xticklabels({'Most unstable nowave','Non wave epochs', 'Wave epochs'})
title('Trial Level Acc.')
signrank(res(:,1),res(:,2))

res=[];
for i=1:size(acc_nonwave,1)
    tmp=squeeze((acc_bin_nonwave(i,:,:)));
    tmp1=squeeze((acc_bin_wave(i,:,:)));
    res(i,:) = [mean(diag(tmp)) mean(diag(tmp1))];
end
figure;boxplot(res)
xticks(1:2)
xticklabels({'Non wave epochs', 'Wave epochs'})
signrank(res(:,1),res(:,2))
title('Bin Level Acc.')

%save hg_wave_nonwave_MLP_3DArrow_CL_AllData_v3_AllFolders acc_nonwave acc_bin_nonwave acc_wave acc_bin_wave -v7.3
%hg_wave_nonwave_MLP for B1 and B6
%hg_wave_nonwave_MLP_3DArrow_CL_v2
%hg_wave_nonwave_MLP_3DArrow_CL_v2_AllData

% v2 -> using smoothed hG in trialdata, all time points (not just during
% correct decodes)

acc_nonwave=squeeze(mean(acc_nonwave,1));
acc_bin_nonwave=squeeze(mean(acc_bin_nonwave,1));


acc_wave=squeeze(mean(acc_wave,1));
acc_bin_wave=squeeze(mean(acc_bin_wave,1));



disp([mean(diag(acc_wave)) mean(diag(acc_bin_wave))])
disp([mean(diag(acc_nonwave)) mean(diag(acc_bin_nonwave))])



%%%%% MATCHING CONTIGUOUS WINDOW BLOCKS %%%%%

% wave
% have to get the data into a cell array of condn_data, with target id and
% neural features
condn_data={};k=1;
for days = 1:length(stats_cl_days)
    stats_cl = stats_cl_days{days};
    stats_cl_hg = stats_cl_hg_days{days};
    for i=1:length(stats_cl)
        tid=stats_cl(i).target_id;
        if tid<=num_targets
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');

            % for j=1:size(tmp,2)
            %     tmp(:,j) =smooth(tmp(:,j),10);
            % end

            % normalize trial length to that of non waves 
            tmp1 = stats_cl_hg(i).hg_nonwave;
            tmp1 = cell2mat(tmp1');

            % if size(tmp,1) > size(tmp1,1)
            %     idx = randperm(size(tmp,1),size(tmp1,1));
            %     %idx = 1:size(tmp1,1);
            %     tmp = tmp(idx,:);
            % end


            % store wave stability values
            % stab = stats_cl(i).stab;
            % [out,st,stp] = wave_stability_detect(zscore(stab));
            % wave_stab=[];
            % for j=1:length(st)
            %     wave_stab=[wave_stab;stab(st(k):stp(k))];
            % end
            condn_data(k).neural = tmp';
            condn_data(k).targetID = tid;
            k=k+1;
        end
    end
end

condn_data1={};k=1;
for i=1:length(condn_data)
    if ~isempty(condn_data(i).neural)
        condn_data1(k).neural = condn_data(i).neural;
        condn_data1(k).targetID = condn_data(i).targetID;
        k=k+1;
    end
end
condn_data=condn_data1;

%% ANALYSIS 4.1 (MAIN) -> PLOTTING RESULTS OF ABOVE

%b1
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/';
cd(root_path)
load hg_wave_nonwave_MLP_3DArrow_CL_v2

res=[];res_bin=[];
for i=1:size(acc_nonwave,1)
    %tmp0=squeeze((acc_nonwave_least(i,:,:)));
    tmp=squeeze((acc_nonwave(i,:,:)));
    tmp1=squeeze((acc_wave(i,:,:)));
    %res(i,:) = [mean(diag(tmp0)) mean(diag(tmp)) mean(diag(tmp1))];
    res(i,:) = [ mean(diag(tmp)) mean(diag(tmp1))];

    tmp=squeeze((acc_bin_nonwave(i,:,:)));
    tmp1=squeeze((acc_bin_wave(i,:,:)));
    res_bin(i,:) = [mean(diag(tmp)) mean(diag(tmp1))];
end
figure;boxplot(100*res)
xticks(1:2)
xticklabels({'Non wave epochs', 'Wave epochs'})
%xticklabels({'Most unstable nowave','Non wave epochs', 'Wave epochs'})
title('B1')
signrank(res(:,1),res(:,2))
plot_beautify
res_B1=res;
ylabel('Trial level Decoding Accuracy')
res_bin_B1= res_bin;

%b3
clearvars -except res_B3 res_B1 res_bin_B1
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
cd(root_path)
load hg_wave_nonwave_MLP_3DArrow_CL_v2

res=[];res_bin=[];
for i=1:size(acc_nonwave,1)
    %tmp0=squeeze((acc_nonwave_least(i,:,:)));
    tmp=squeeze((acc_nonwave(i,:,:)));
    tmp1=squeeze((acc_wave(i,:,:)));
    %res(i,:) = [mean(diag(tmp0)) mean(diag(tmp)) mean(diag(tmp1))];
    res(i,:) = [ mean(diag(tmp)) mean(diag(tmp1))];

    tmp=squeeze((acc_bin_nonwave(i,:,:)));
    tmp1=squeeze((acc_bin_wave(i,:,:)));
    res_bin(i,:) = [mean(diag(tmp)) mean(diag(tmp1))];
end
figure;boxplot(100*res,'Whisker',2)
xticks(1:2)
xticklabels({'Non wave epochs', 'Wave epochs'})
%xticklabels({'Most unstable nowave','Non wave epochs', 'Wave epochs'})
title('B3')
signrank(res(:,1),res(:,2))
plot_beautify
res_B3=res;
ylabel('Trial level Decoding Accuracy')
res_bin_B3= res_bin;

%b6
clearvars -except res_B3 res_B1 res_bin_B3 res_bin_B1
root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6';
cd(root_path)
load hg_wave_nonwave_MLP_3DArrow_CL_AllData_v3_AllFolders

res=[];res_bin=[];
for i=1:size(acc_nonwave,1)
    %tmp0=squeeze((acc_nonwave_least(i,:,:)));
    tmp=squeeze((acc_nonwave(i,:,:)));
    tmp1=squeeze((acc_wave(i,:,:)));
    %res(i,:) = [mean(diag(tmp0)) mean(diag(tmp)) mean(diag(tmp1))];
    res(i,:) = [ mean(diag(tmp)) mean(diag(tmp1))];

    tmp=squeeze((acc_bin_nonwave(i,:,:)));
    tmp1=squeeze((acc_bin_wave(i,:,:)));
    res_bin(i,:) = [mean(diag(tmp)) mean(diag(tmp1))];
end
figure;boxplot(100*res,'Whisker',2)
xticks(1:2)
xticklabels({'Non wave epochs', 'Wave epochs'})
%xticklabels({'Most unstable nowave','Non wave epochs', 'Wave epochs'})
title('B6')
signrank(res(:,1),res(:,2))
plot_beautify
res_B6=res;
ylabel('Trial level Decoding Accuracy')
res_bin_B6=res_bin;


res_bin =[res_bin_B1;res_bin_B3;res_bin_B6];
%res_bin =[res_bin_B6];
figure;
boxplot(res_bin,'notch','on')
ylim([0.475 0.625])
signrank(res_bin(:,1),res_bin(:,2))


%% ANALYSIS 4.5 WITHIN DAY VERSION OF ABOVE
% build a classifer across days looking at decoding performance in hg
% during wave vs. non wave epochs

% have to get the data into a cell array of condn_data, with target id and
% neural features

iterations=10;
num_targets=7;
acc_wave_days=[];
acc_nonwave_days=[];
for days = 1:length(stats_cl_days)
    disp(['Processing Day ' num2str(days)])
    stats_cl = stats_cl_days{days};
    stats_cl_hg = stats_cl_hg_days{days};

    % during wave epochs
    condn_data={};k=1;
    for i=1:length(stats_cl)
        tid=stats_cl(i).target_id;
        if tid<=num_targets+1
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');
            for j=1:size(tmp,2)
                tmp(:,j) =smooth(tmp(:,j),15);
            end
            condn_data(k).neural = tmp';
            condn_data(k).targetID = tid;
            k=k+1;
        end
    end

    condn_data1={};k=1;
    for i=1:length(condn_data)
        if ~isempty(condn_data(i).neural)
            condn_data1(k).neural = condn_data(i).neural;
            condn_data1(k).targetID = condn_data(i).targetID;
            k=k+1;
        end
    end
    condn_data=condn_data1;
    
    [acc_wave,train_permutations,acc_bin_wave,bino_pdf,bino_pdf_chance] = ...
        accuracy_imagined_data_8DoF(condn_data, iterations);
    acc_wave = squeeze(nanmean(acc_wave,1));
    acc_wave_days(:,days) = diag(acc_wave);


    % during non wave epochs
    condn_data={};k=1;
    for i=1:length(stats_cl)
        tid=stats_cl(i).target_id;
        if tid<=num_targets+1
            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');
            for j=1:size(tmp,2)
                tmp(:,j) =smooth(tmp(:,j),15);
            end
            condn_data(k).neural = tmp';
            condn_data(k).targetID = tid;
            k=k+1;
        end
    end

    condn_data1={};k=1;
    for i=1:length(condn_data)
        if ~isempty(condn_data(i).neural)
            condn_data1(k).neural = condn_data(i).neural;
            condn_data1(k).targetID = condn_data(i).targetID;
            k=k+1;
        end
    end
    condn_data=condn_data1;

    [acc_nonwave,train_permutations,acc_bin_nonwave,bino_pdf,bino_pdf_chance] = ...
    accuracy_imagined_data_8DoF(condn_data, iterations);
    acc_nonwave = squeeze(nanmean(acc_nonwave,1));
    acc_nonwave_days(:,days) = diag(acc_nonwave);

end

figure;plot(median(acc_wave_days))
hold on
plot(median(acc_nonwave_days))
legend('Waves','Nonwaves')

x=[1:10]';
y = [median(acc_nonwave_days)]';
[bhat p wh se ci t_stat]=robust_fit(x,y,2);p


%% ANALYSIS 4.9 (MAIN STORING DATA FOR WAVE EPOCHS AND AE TO UNDERSTAND
% REPRESENTATIONAL GEOMETRY OF WAVE AND NON WAVE EPOCHS). 

%%%% STORING SINGLE TRIAL DATA FOR WAVES AND NON WAVES IN CL
condn_data={};k=1;
for days = 1:length(stats_cl_days)
    stats_cl = stats_cl_days{days};
    stats_cl_hg = stats_cl_hg_days{days};
    for i=1:length(stats_cl)
        tid=stats_cl(i).target_id;
        acc=stats_cl(i).accuracy;
        if tid<=num_targets

            % day information
            condn_data(k).day = days;

            % trial number
            condn_data(k).trial = k;

            % target information
            condn_data(k).targetID = tid;

            % trial accuracy information
            condn_data(k).accuracy = acc;

            % store wave data
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');
            % smoothing for B6 and B1 alone
            % if strcmp(subj,'B1') || strcmp(subj,'B6')
            %     for j=1:size(tmp,2)
            %         tmp(:,j) =smooth(tmp(:,j),10);
            %     end
            % end
            condn_data(k).wave_neural = tmp';

            % store nonwave data
            tmp = stats_cl_hg(i).hg_nonwave;
            tmp = cell2mat(tmp');
            % smoothing for B6 and B1 alone
            % if strcmp(subj,'B1') || strcmp(subj,'B6')
            %     for j=1:size(tmp,2)
            %         tmp(:,j) =smooth(tmp(:,j),10);
            %     end
            % end
            condn_data(k).nonwave_neural = tmp';

            % store wave stability values
            % stab = stats_cl(i).stab;
            % [out,st,stp] = wave_stability_detect(zscore(stab));
            % wave_stab=[];
            % for j=1:length(st)
            %     wave_stab=[wave_stab;stab(st(k):stp(k))];
            % end


            k=k+1;
        end
    end
end


% save
%save B3_Wave_NonWave_hG_For_AE condn_data -v7.3 
%save B1_Wave_NonWave_hG_For_AE condn_data -v7.3 
save B6_Wave_NonWave_hG_For_AE condn_data -v7.3 

len=[];len1=[];
for i=1:length(condn_data)
    tmp=condn_data(i).wave_neural;
    len=[len;size(tmp,2)];
    tmp=condn_data(i).nonwave_neural;
    len1=[len1;size(tmp,2)];
end

disp('Section 4.9 finished running')

% % remove trials without any wave or non wave data (OPTIONAL)
% condn_data1={};k=1;
% for i=1:length(condn_data)
%     if ~isempty(condn_data(i).wave_neural)
%         condn_data1(k).neural = condn_data(i).wave_neural;
%         condn_data1(k).targetID = condn_data(i).targetID;
%         k=k+1;
%     end
% end
% condn_data=condn_data1;

%% ANALYSIS 5   (MAIN, IMPT)
% duty cycle within accurate and inaccurate trials

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
load('B1_waves_stability_hgFilterBank_PLV_AccStatsCL_v2.mat','stats_cl_days')
[res_days_B1, res_days_f_B1, res_days_d_B1] = get_duty_cycle(stats_cl_days);

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
load('B3_waves_3DArrow_stability_hgFilterBank_PLV_AccStatsCL_v2.mat','stats_cl_days')
[res_days_B3, res_days_f_B3, res_days_d_B3] = get_duty_cycle(stats_cl_days);

cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
load('B6_waves_stability_hgFilterBank_PLV_AccStatsCL_v2_AllData.mat','stats_cl_days')
[res_days_B6, res_days_f_B6, res_days_d_B6] = get_duty_cycle(stats_cl_days);

% res_days_B1 = log(res_days_B1);
% res_days_B3 = log(res_days_B3);
% res_days_B6 = log(res_days_B6);

res_days = [res_days_B1;res_days_B3;res_days_B6];
figure;hold on
%b1
idx = [ones(size(res_days_B1,1),1) 2*ones(size(res_days_B1,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B1(:,1),50,'b','LineWidth',1)
scatter(idx(:,2),res_days_B1(:,2),50,'b',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B1(i,1),res_days_B1(i,2)],...
        'Color',[0 0 1 .25])
end

%b3
%figure;hold on
idx = [ones(size(res_days_B3,1),1) 2*ones(size(res_days_B3,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B3(:,1),50,'r','LineWidth',1)
scatter(idx(:,2),res_days_B3(:,2),50,'r',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B3(i,1),res_days_B3(i,2)],...
        'Color',[1 0 0 .25])
end

%b6
%figure;hold on
idx = [ones(size(res_days_B6,1),1) 2*ones(size(res_days_B6,1),1)];
idx = idx + 0.05*randn(size(idx));
scatter(idx(:,1),res_days_B6(:,1),50,'k','LineWidth',1)
scatter(idx(:,2),res_days_B6(:,2),50,'k',"filled",'LineWidth',1)
for i=1:size(idx,1)
    plot([idx(i,1),idx(i,2)],[res_days_B6(i,1),res_days_B6(i,2)],...
        'Color',[0.25 0.25 0.25 .25])
end
ylim([0.32 0.46])
xlim([.5 2.5])
xticks(1:2)
xticklabels({'Inaccurate Trials','Accurate Trials'})
ylabel('Duty Cycle')
plot_beautify

[p,h] = signrank(res_days(:,1),res_days(:,2))

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


