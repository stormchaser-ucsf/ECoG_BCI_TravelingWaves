function res_days = get_plv_waves_singleTrials(stats_cl_hg_days)



parallel_check
res_days=[];
for days=1:length(stats_cl_hg_days)

    stats = stats_cl_hg_days{days};
    wave_plv_trial=[];
    nonwave_plv_trial=[];
    res=[];
    for i =1:length(stats)
        a=stats(i).plv_wave; %already phase difference, exp(1i*delta theta)
        b = stats(i).plv_nonwave; %already phase difference, exp(1i*delta theta)
        wave_len_cl(i) = size(a,1);
        nonwave_len_cl(i) = size(b,1);

        wave_plv=[];
        nonwave_plv=[];


        parfor iter=1:50


            if wave_len_cl(i)<nonwave_len_cl(i)
                wave_plv(iter,:) = (abs(mean(a)));%get coupling strength single trial

                len = min(30,nonwave_len_cl(i) -  wave_len_cl(i));
                idx=randperm(nonwave_len_cl(i) -  wave_len_cl(i),len);
                plv_tmp=[];
                for j=1:1%length(idx)
                    tmp = b(idx(j):idx(j)+wave_len_cl(i)-1,:);
                    plv_tmp(j,:) = (abs(mean(tmp,1)));%get coupling strength single trial
                end
                nonwave_plv(iter,:) = (plv_tmp);
                %nonwave_plv(i,:,:) = (plv_tmp);

            elseif wave_len_cl(i)>nonwave_len_cl(i)
                %nonwave_plv(iter,:) = (angle(mean(b))); %get preferred angle
                nonwave_plv(iter,:) = (abs(mean(b))); %get coupling strength single trial

                len = min(30,wave_len_cl(i) -  nonwave_len_cl(i));
                idx=randperm(wave_len_cl(i) -  nonwave_len_cl(i),len);
                plv_tmp=[];
                for j=1:1%length(idx)
                    tmp = a(idx(j):idx(j)+nonwave_len_cl(i)-1,:);
                    %plv_tmp(j,:) = (angle(mean(tmp,1)));%get preferred angle
                    plv_tmp(j,:) = (abs(mean(tmp,1)));%get coupling strength single trial
                end
                wave_plv(iter,:) = (plv_tmp);
                %wave_plv(i,:,:) = (plv_tmp);

            elseif wave_len_cl(i)== nonwave_len_cl(i)
                %nonwave_plv(iter,:) = (angle(mean(b)));
                %wave_plv(iter,:) = (angle(mean(a)));
                nonwave_plv(iter,:) = (abs(mean(b)));
                wave_plv(iter,:) = (abs(mean(a)));
            end

        end
        wave_plv = (mean(wave_plv,1));
        nonwave_plv = (mean(nonwave_plv,1));

        wave_plv_trial(i) = mean(wave_plv);
        nonwave_plv_trial(i) = mean(nonwave_plv);

    end
    res = [wave_plv_trial' nonwave_plv_trial'];
    res_days(days,:) = mean(res);
end


figure;
boxplot(res_days)
[p,h]=signrank(res_days(:,1), res_days(:,2));
title(['pval of ' num2str(p)])
xticks(1:2)
xticklabels({'Wave epochs','Non wave epochs'})
ylabel('Single Trial PLV')
plot_beautify

