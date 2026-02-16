function [res_days] = get_plv_waves(stats_cl_hg_days)





%%%%%% JUST FOR CLOSED LOOP, TRIAL LENGTH MATCHING
elec_list=[34	162	165	168	172	176
51	47	43	39	36	33
219	223	94	90	86	83
203	207	210	213	216	220
76	72	68	196	200	204];

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

% %%%NEW METHOD
% traveling wave enforces consistency in phase relationship across trials
% unlike non traveling wave epochs
res_days=[];
parfor days=1:length(stats_cl_hg_days)
    stats_cl_hg = stats_cl_hg_days{days};
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

        wave_plv(i,:) = angle(mean(a,1));
        nonwave_plv(i,:) = angle(mean(b,1));



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

    end

    wave_plv = exp(1i*wave_plv);
    wave_plv = abs(mean(wave_plv,1));

    nonwave_plv = exp(1i*nonwave_plv);
    nonwave_plv = abs(mean(nonwave_plv,1));

    res_days(days,:) = [mean(wave_plv) mean(nonwave_plv)];
end


% %%%OLD METHOD
% res_days=[];
% parfor days=1:length(stats_cl_hg_days)
%     stats_cl_hg = stats_cl_hg_days{days};
%     wave_len_cl=[];
%     nonwave_len_cl=[];
%     wave_plv=[];
%     nonwave_plv=[];
%     for i=1:length(stats_cl_hg)
%         %%% just straight up average plv across grid
%         a=stats_cl_hg(i).plv_wave;
%         %a=a(:,elec_list);
%         wave_len_cl(i) = size(a,1);
% 
%         b = stats_cl_hg(i).plv_nonwave;
%         %b=b(:,elec_list);
%         nonwave_len_cl(i) = size(b,1);
% 
%         % nonwave_plv(i) = mean(abs(mean(b)));
%         % wave_plv(i) = mean(abs(mean(a)));
% 
%         if wave_len_cl(i)<nonwave_len_cl(i)
%             wave_plv(i) = mean(abs(mean(a)));
% 
%             len = min(30,nonwave_len_cl(i) -  wave_len_cl(i));
%             idx=randperm(nonwave_len_cl(i) -  wave_len_cl(i),len);
%             plv_tmp=[];
%             for j=1:length(idx)
%                 tmp = b(idx(j):idx(j)+wave_len_cl(i)-1,:);
%                 plv_tmp(j) = mean(abs(mean(tmp,1)));
%             end
%             nonwave_plv(i) = mean(plv_tmp);
% 
%         elseif wave_len_cl(i)>nonwave_len_cl(i)
%             nonwave_plv(i) = mean(abs(mean(b)));
% 
%             len = min(30,wave_len_cl(i) -  nonwave_len_cl(i));
%             idx=randperm(wave_len_cl(i) -  nonwave_len_cl(i),len);
%             plv_tmp=[];
%             for j=1:length(idx)
%                 tmp = a(idx(j):idx(j)+nonwave_len_cl(i)-1,:);
%                 plv_tmp(j) = mean(abs(mean(tmp,1)));
%             end
%             wave_plv(i) = mean(plv_tmp);
% 
%         elseif wave_len_cl(i)== nonwave_len_cl(i)
%             nonwave_plv(i) = mean(abs(mean(b)));
%             wave_plv(i) = mean(abs(mean(a)));
%         end
%     end
%     res_days(days,:) = [median(wave_plv) median(nonwave_plv)];
% end
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

