function [res_days] = get_plv_waves(stats_cl_hg_days,ecog_grid,cortex,elecmatrix)


%%%%%% JUST FOR CLOSED LOOP, TRIAL LENGTH MATCHING
elec_list=[34	162	165	168	172	176	180
51	47	43	39	36	33	161
219	223	94	90	86	83	80
203	207	210	213	216	220	224
76	72	68	196	200	204	237];

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
res_days_map=[];
for days=1:length(stats_cl_hg_days)
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
    res_days_map(days,:,1) = wave_plv;
    res_days_map(days,:,2) = nonwave_plv;
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




% plotting cortical channels
sig = res_days_map(:,:,1) - res_days_map(:,:,2);
sig = mean(sig,1);
%a = squeeze(median(res_days_map(1:end,:,1),1));
%b = squeeze(median(res_days_map(1:end,:,2),1));
%sig = a;
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
        if sig1(j)==1 && ch_wts(j)<0
            ms = abs(ch_wts(j))*150;
            %[aa bb]=min(abs(pref_phase(j) - phMap));
            %c=ChColorMap(bb,:);
            e_h = el_add(elecmatrix(ch,:), 'color', 'b','msize',ms);
        end
    end
end

set(gcf,'Color','w')
sgtitle('Channels with greater increase in hG variance due to mu')
