function [res_days,res_days_f,res_days_d] = get_duty_cycle(stats_cl_days)


res_days=[];
res_days_f=[];
res_days_d=[];
for days=1:length(stats_cl_days)
    res_acc=[];dc_acc=[];d_acc=[];f_acc=[];
    res_err=[];dc_err=[];d_err=[];f_err=[];
    stats_cl = stats_cl_days{days};
    for i=1:length(stats_cl)
        stab = zscore(stats_cl(i).stab);
        [out,st,stp]=wave_stability_detect(stab);
        wav_det=zeros(length(stab),1);
        for k=1:length(st)
            wav_det(st(k):stp(k))=1;
        end

        output = stats_cl(i).output;
        idx=find(output==1);
        stab_acc = wav_det(idx);
        prop_waves = sum(wav_det(idx))/length(idx);

        % if isnan(prop_waves)
        %     prop_waves=0;
        % end

        % duty cycle
        tmp=stab;
        t = length(tmp) * 20/1e3;
        f =length(out)/t; % frequency/s
        d = mean(out) * 20/1e3; %duration in s
        dcyc=f*d;



        if stats_cl(i).accuracy==1
            res_acc = [res_acc;prop_waves ];
            dc_acc = [dc_acc;dcyc];
            d_acc = [d_acc;d];
            f_acc = [f_acc;f];
        else
            res_err = [res_err;prop_waves ];
            dc_err = [dc_err;dcyc];
            d_err = [d_err;d];
            f_err = [f_err;f];
        end
    end
    res_days(days,:)=[mean(dc_err) mean(dc_acc)];
    %res_days(days,:)=[mean(res_err) mean(res_acc)];
    res_days_f(days,:) = [mean(f_err) mean(f_acc)];
    res_days_d(days,:) = [mean(d_err) mean(d_acc)];
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
