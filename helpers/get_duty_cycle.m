function [res_days,res_days_f,res_days_d] = ...
    get_duty_cycle(stats_cl_days,stats_cl_hg_days)


res_days=[];
res_days_f=[];
res_days_d=[];
res_days_pow=[];
ch=[137	143	148	152
159	160	30	28
134	140	170	174
49	45	41	38
221	62	59	56];
for days=1:length(stats_cl_days)
    stats_cl = stats_cl_days{days};
    res_acc=[];dc_acc=[];d_acc=[];f_acc=[];
    res_err=[];dc_err=[];d_err=[];f_err=[];
    stats_cl_hg = stats_cl_hg_days{days};
    mu_pow_acc=[];mu_pow_err=[];
    for i=1:length(stats_cl)
        %%% old
        %stab = zscore(stats_cl(i).stab);
        %[out,st,stp]=wave_stability_detect(stab);


        %%% new, after state 1 roughly
        stab = stats_cl(i).stab;
        l=length(stab);
        % if l>=50
        % 
        %     stab = stab(1:50);% duration matching
        % end

        stab1 = zscore(stab(15:end));
        [out,st,stp] = wave_stability_detect(stab1);
        st = st+14;
        stp = stp+14;

        wav_det=zeros(length(stab),1);
        for k=1:length(st)
            wav_det(st(k):stp(k))=1;
        end

        output = stats_cl(i).output;
        % if l>=50
        %     output = output(1:50);% duration matching
        % end

        idx=find(output==1);
        stab_acc = wav_det(idx);
        prop_waves = sum(wav_det(idx))/length(idx);

        % if isnan(prop_waves)
        %     prop_waves=0;
        % end

        % duty cycle
        tmp=stab1;
        t = length(tmp) * 20/1e3;
        f = length(out)/t; % frequency/s
        d = mean(out) * 20/1e3; %duration in s
        dcyc=f*d;

        % get mu power during waves
        tmp = stats_cl_hg(i).mu_wave;
        tmp = cell2mat(tmp');
        tmp = tmp(:,ch);
        tmp = mean(abs(tmp(:)));




        if stats_cl(i).accuracy==1
            res_acc = [res_acc;prop_waves ];
            dc_acc = [dc_acc;dcyc];
            d_acc = [d_acc;d];
            f_acc = [f_acc;f];
            mu_pow_acc = [mu_pow_acc;tmp];
        else
            res_err = [res_err;prop_waves ];
            dc_err = [dc_err;dcyc];
            d_err = [d_err;d];
            f_err = [f_err;f];
            mu_pow_err = [mu_pow_err;tmp];
        end
    end
    %[nanmean(dc_err) nanmean(dc_acc)]
    res_days(days,:)=[nanmean(dc_err) nanmean(dc_acc)];
    %res_days(days,:)=[mean(res_err) mean(res_acc)];
    res_days_f(days,:) = [nanmean(f_err) nanmean(f_acc)];
    res_days_d(days,:) = [nanmean(d_err) nanmean(d_acc)];
    res_days_pow(days,:) = [nanmean(mu_pow_err) nanmean(mu_pow_acc)];
end

[p,h]=signrank(res_days(:,1),res_days(:,2))
[ h p tb st]=ttest(res_days(:,1),res_days(:,2))
figure;boxplot(res_days)
xticks(1:2)
xticklabels({'Error Trials','Acc. Trails'})

figure;
plot3(res_days_f(:,1),res_days_d(:,1),res_days(:,1),'.r','MarkerSize',15)
hold on
plot3(res_days_f(:,2),res_days_d(:,2),res_days(:,2),'.b','MarkerSize',15)
xlabel('Freq')
ylabel('Mean duration')
zlabel('Duty Cycle')
grid on
view(17,31)
%
%[az, el] = view;
% fprintf('Azimuth: %.2f, Elevation: %.2f\n', az, el);


figure;
boxplot(res_days_pow)
xticks(1:2)
xticklabels({'Error Trials','Acc. Trails'})
ylabel('Mu power during waves')
[p,h]=signrank(res_days_pow(:,1),res_days_pow(:,2))
