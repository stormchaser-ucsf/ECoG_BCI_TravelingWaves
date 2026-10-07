function res = get_plv_stats_intact(stats,chc)

if nargin<2
    chc=1;
end

if chc==1 % single trial analyses plv magnitude

    wave_plv_trial=[];
    nonwave_plv_trial=[];
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
    figure;
    boxplot(res)
    [p,h]=signrank(wave_plv_trial, nonwave_plv_trial);
    title(['pval of ' num2str(p)])
    xticks(1:2)
    xticklabels({'Wave epochs','Non wave epochs'})
    ylabel('PLV')
    plot_beautify

end


if chc==2 % across trial consistency in preferred phase angle
    wave_plv_iter=[];
    nonwave_plv_iter=[];
    parfor iter=1:20
        wave_len_cl=[];
        nonwave_len_cl=[];
        wave_plv=[];
        nonwave_plv=[];
        for i=1:length(stats)
            %%% just straight up average plv across grid
            a=stats(i).plv_wave;
            %a=a(:,elec_list);
            wave_len_cl(i) = size(a,1);

            b = stats(i).plv_nonwave;
            %b=b(:,elec_list);
            nonwave_len_cl(i) = size(b,1);

            % nonwave_plv(i) = mean(abs(mean(b)));
            % wave_plv(i) = mean(abs(mean(a)));

            % wave_plv(i,:) = angle(mean(a,1));
            % nonwave_plv(i,:) = angle(mean(b,1));



            if wave_len_cl(i)<nonwave_len_cl(i)
                wave_plv(i,:) = (angle(mean(a)));

                len = min(30,nonwave_len_cl(i) -  wave_len_cl(i));
                idx=randperm(nonwave_len_cl(i) -  wave_len_cl(i),len);
                plv_tmp=[];
                for j=1:1%length(idx)
                    tmp = b(idx(j):idx(j)+wave_len_cl(i)-1,:);
                    plv_tmp(j,:) = (angle(mean(tmp,1)));
                end
                nonwave_plv(i,:) = circ_mean(plv_tmp);
                %nonwave_plv(i,:,:) = (plv_tmp);

            elseif wave_len_cl(i)>nonwave_len_cl(i)
                nonwave_plv(i,:) = (angle(mean(b)));

                len = min(30,wave_len_cl(i) -  nonwave_len_cl(i));
                idx=randperm(wave_len_cl(i) -  nonwave_len_cl(i),len);
                plv_tmp=[];
                for j=1:1%length(idx)
                    tmp = a(idx(j):idx(j)+nonwave_len_cl(i)-1,:);
                    plv_tmp(j,:) = (angle(mean(tmp,1)));
                end
                wave_plv(i,:) = circ_mean(plv_tmp);
                %wave_plv(i,:,:) = (plv_tmp);

            elseif wave_len_cl(i)== nonwave_len_cl(i)
                nonwave_plv(i,:) = (angle(mean(b)));
                wave_plv(i,:) = (angle(mean(a)));
            end

        end
        wave_plv = exp(1i*wave_plv);
        wave_plv = abs(mean(wave_plv,1));
        wave_plv_iter(iter,:)= wave_plv;


        nonwave_plv = exp(1i*nonwave_plv);
        nonwave_plv = abs(mean(nonwave_plv,1));
        nonwave_plv_iter(iter,:)= nonwave_plv;
    end


    wave_plv = median(wave_plv_iter,1);
    nonwave_plv = median(nonwave_plv_iter,1);


    res = [wave_plv' nonwave_plv'];
    figure;
    boxplot(res)
    xticks(1:2)
    xticklabels({'Wave epochs','Non wave epochs'})
    plot_beautify
    [p,h] = signrank(res(:,1),res(:,2));
    title(num2str(p))
end





%
% parfor iter=1:20
%     wave_len_cl=[];
%     nonwave_len_cl=[];
%     wave_plv=[];
%     nonwave_plv=[];
%     for i=1:length(stats)
%         %%% just straight up average plv across grid
%         a=stats(i).plv_wave;
%         %a=a(:,elec_list);
%         wave_len_cl(i) = size(a,1);
%
%         b = stats(i).plv_nonwave;
%         %b=b(:,elec_list);
%         nonwave_len_cl(i) = size(b,1);
%
%         % nonwave_plv(i) = mean(abs(mean(b)));
%         % wave_plv(i) = mean(abs(mean(a)));
%
%         % wave_plv(i,:) = angle(mean(a,1));
%         % nonwave_plv(i,:) = angle(mean(b,1));
%
%
%
%         if wave_len_cl(i)<nonwave_len_cl(i)
%             wave_plv(i,:) = (angle(mean(a)));
%
%             len = min(30,nonwave_len_cl(i) -  wave_len_cl(i));
%             idx=randperm(nonwave_len_cl(i) -  wave_len_cl(i),len);
%             plv_tmp=[];
%             for j=1:1%length(idx)
%                 tmp = b(idx(j):idx(j)+wave_len_cl(i)-1,:);
%                 plv_tmp(j,:) = (angle(mean(tmp,1)));
%             end
%             nonwave_plv(i,:) = circ_mean(plv_tmp);
%             %nonwave_plv(i,:,:) = (plv_tmp);
%
%         elseif wave_len_cl(i)>nonwave_len_cl(i)
%             nonwave_plv(i,:) = (angle(mean(b)));
%
%             len = min(30,wave_len_cl(i) -  nonwave_len_cl(i));
%             idx=randperm(wave_len_cl(i) -  nonwave_len_cl(i),len);
%             plv_tmp=[];
%             for j=1:1%length(idx)
%                 tmp = a(idx(j):idx(j)+nonwave_len_cl(i)-1,:);
%                 plv_tmp(j,:) = (angle(mean(tmp,1)));
%             end
%             wave_plv(i,:) = circ_mean(plv_tmp);
%             %wave_plv(i,:,:) = (plv_tmp);
%
%         elseif wave_len_cl(i)== nonwave_len_cl(i)
%             nonwave_plv(i,:) = (angle(mean(b)));
%             wave_plv(i,:) = (angle(mean(a)));
%         end
%
%     end
%     wave_plv = exp(1i*wave_plv);
%     wave_plv = abs(mean(wave_plv,1));
%     wave_plv_iter(iter,:)= wave_plv;
%
%
%     nonwave_plv = exp(1i*nonwave_plv);
%     nonwave_plv = abs(mean(nonwave_plv,1));
%     nonwave_plv_iter(iter,:)= nonwave_plv;
% end
%
%
% wave_plv = mean(wave_plv_iter,1);
% nonwave_plv = mean(nonwave_plv_iter,1);
