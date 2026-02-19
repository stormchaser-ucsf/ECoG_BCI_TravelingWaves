function [res_days] = get_Channel_TrialVariance(stats_cl_hg_days,num_targets)


res_days=[];pval=[];
res_std_days=[];
res_days_map=[];
for days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    res=[];
    res_std=[];
    res_map=[];
    for tid=1:num_targets
        D_wave=[];D_nonwave=[];
        D_wave_total=[];D_nonwave_total=[];
        for i=1:length(stats_cl_hg)
            if stats_cl_hg(i).target_id ==tid
                tmp = stats_cl_hg(i).hg_wave;
                tmp = cell2mat(tmp');tmp1=tmp;

                idx = randperm(size(tmp,1),20);
                tmp = tmp(idx,:);

                tmp = mean(tmp,1);
                if ~isempty(tmp) && sum(~isnan(tmp)) == 253
                    D_wave = cat(1,D_wave,tmp);
                    D_wave_total = [D_wave_total;tmp1];
                end

                tmp = stats_cl_hg(i).hg_nonwave;
                tmp = cell2mat(tmp');tmp1=tmp;

                idx = randperm(size(tmp,1),20);
                tmp = tmp(idx,:);

                tmp = mean(tmp,1); % or mean here or var
                if ~isempty(tmp) && sum(~isnan(tmp)) == 253
                    D_nonwave = cat(1,D_nonwave,tmp);
                    D_nonwave_total = [D_nonwave_total;tmp1];
                end
            end
        end
        tmp=log([(std(D_wave,1))' (std(D_nonwave,1))']);
        res_map = [res_map tmp];
        res = [res;median(tmp,1)];
    end
    %res = ([(var(D_wave,1))' (var(D_nonwave,1))']);
    
    %res_std = [(log(std(D_wave_total,1)))' (log(std(D_nonwave_total,1)))'];
    % figure;
    % boxplot(res)
    %[p,h] = signrank(res(:,1),res(:,2));
    %pval(days) = ((p<0.05) * (median(res(:,1)) - median(res(:,2))));
    % xticks(1:2)
    % xticklabels({'Wave epochs','Non wave epochs'})
    % ylabel('Variability in mean activity across conditions')
    
    res_days =[res_days ;mean(res,1)];
    %res_days =[res_days ;res];

    res_days_map(days,:,1) = mean(res_map(:,1:2:end),2);
    res_days_map(days,:,2) = mean(res_map(:,2:2:end),2);


    %res_std_days=[res_std_days;median(res_std,1)];
    %res_days(days,:,:) = res;
end
[p,h] = signrank(res_days(:,1),res_days(:,2))
% mean(res_days)
figure;boxplot(res_days,'Notch','off')
% idx=[0.01*randn(size(res_days,1),1) + ones(size(res_days,1),1)...
%     0.01*randn(size(res_days,1),1) + 2*ones(size(res_days,1),1)];
% figure;scatter(idx,res_days)
% xlim([0.5 2.5])
% hold on
% for i=1:size(res_days,1)
%     plot([idx(i,1) idx(i,2)],[res_days(i,1) res_days(i,2)],'Color',[.5 .5 .5 .5]);
% end
% res_days_cl=res_days;
% xticks(1:2)
xticklabels({'Wave','Non wave'})
ylabel('Trial to Trial Variance')


% plotting cortical channels
a = squeeze(mean(res_days_map(:,:,1),1));
b = squeeze(mean(res_days_map(:,:,2),1));
sig = a-b;
sig1 = [sig(1:107) 0 sig(108:111) 0  sig(112:115) 0 ...
    sig(116:end)];
figure;imagesc(sig1(ecog_grid))
ch_wts = sig1;

sig = ones(253,1)';
sig1 = [sig(1:107) 0 sig(108:111) 0  sig(112:115) 0 ...
    sig(116:end)];

ch_layout=[];
for i=1:23:253
    ch_layout = [ch_layout; i:i+22 ];
end
ch_layout = (fliplr(ch_layout));

%plv(sig) = zscore(plv(sig))+4;
phMap = linspace(-pi,pi,253)';
ChColorMap = ([parula(253)]);
figure
%subplot(1,2,2)
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix, 'color', 'w','msize',2);
% elecmatrix1 = [elecmatrix(1:107,:); zeros(1,3); elecmatrix(108:111,:); zeros(1,3) ; ...
%     elecmatrix(112:115,:) ;zeros(1,3); elecmatrix(116:end,:)];
for j=1:256%length(sig)
    [xx yy]=find(ecog_grid==j);
    if ~isempty(xx)
        ch = ch_layout(xx,yy);
        if sig1(j)==1 && ch_wts(j)>0
            ms = ch_wts(j)*30;
            %[aa bb]=min(abs(pref_phase(j) - phMap));
            c=ChColorMap(bb,:);
            e_h = el_add(elecmatrix(ch,:), 'color', 'b','msize',ms);
        end
    end
end

set(gcf,'Color','w')
sgtitle('Channels with greater increase in hG variance due to mu')


