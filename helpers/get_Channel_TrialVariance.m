function [res_days] = get_Channel_TrialVariance(stats_cl_hg_days,num_targets)


res_days=[];pval=[];
res_std_days=[];
for days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
    res=[];
    res_std=[];
    for tid=1:num_targets
        D_wave=[];D_nonwave=[];
        D_wave_total=[];D_nonwave_total=[];
        for i=1:length(stats_cl_hg)
            if stats_cl_hg(i).target_id ==tid
                tmp = stats_cl_hg(i).hg_wave;
                tmp = cell2mat(tmp');tmp1=tmp;

                % idx = randperm(size(tmp,1),20);
                % tmp = tmp(idx,:);

                tmp = mean(tmp,1);
                if ~isempty(tmp) && sum(~isnan(tmp)) == 253
                    D_wave = cat(1,D_wave,tmp);
                    D_wave_total = [D_wave_total;tmp1];
                end

                tmp = stats_cl_hg(i).hg_nonwave;
                tmp = cell2mat(tmp');tmp1=tmp;

                % idx = randperm(size(tmp,1),20);
                % tmp = tmp(idx,:);

                tmp = mean(tmp,1); % or mean here or var
                if ~isempty(tmp) && sum(~isnan(tmp)) == 253
                    D_nonwave = cat(1,D_nonwave,tmp);
                    D_nonwave_total = [D_nonwave_total;tmp1];
                end
            end
        end
        tmp=log([(std(D_wave,1))' (std(D_nonwave,1))']);
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



