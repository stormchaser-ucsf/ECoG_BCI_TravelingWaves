function res = get_wave_summary_stats(stats_cl_days)


wave_duration_days=[];
num_waves_days=[];num_waves_all=[];
duty_cycle_days=[];
for days=1:length(stats_cl_days)    
    stats_cl = stats_cl_days{days};
    wave_dur=[];
    num_waves=[];
    duty_cycle=[];
    for i=1:length(stats_cl)

        stab = stats_cl(i).stab;
        stab1 = zscore(stab(15:end));
        [out,st,stp] = wave_stability_detect(stab1);
        st = st+14;
        stp = stp+14;
        wave_dur = [wave_dur out*20];
        num_waves = [num_waves length(out)/(length(stab1)*20/1e3)];

        tmp=stab1;
        t = length(tmp) * 20/1e3;
        f = length(out)/t; % frequency/s
        d = mean(out) * 20/1e3; %duration in s
        duty_cycle = [duty_cycle f*d];        
    end
    wave_duration_days(days) = mean(wave_dur);
    num_waves_days(days) = mean(num_waves);
    duty_cycle_days(days) = mean(duty_cycle);
    num_waves_all = [num_waves_all num_waves];
end
%figure;boxplot(wave_duration_days)
% figure;boxplot(num_waves_days)
%figure;boxplot(duty_cycle_days)
res.wave_duration_days = wave_duration_days;
res.num_waves_days = num_waves_days;
res.duty_cycle_days = duty_cycle_days;

