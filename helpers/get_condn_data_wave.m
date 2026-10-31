function condn_data = get_condn_data_wave(stats_cl_hg_days,stats_cl_days,bad_ch,num_targets)

condn_data={};k=1;
for days = 1:length(stats_cl_days)
    stats_cl = stats_cl_days{days};
    stats_cl_hg = stats_cl_hg_days{days};
    for i=1:length(stats_cl)
        tid=stats_cl(i).target_id;
        if tid<=num_targets
            tmp = stats_cl_hg(i).hg_wave;
            tmp = cell2mat(tmp');

            for j=1:size(tmp,2)
                tmp(:,j) =smooth(tmp(:,j),10);
            end



            % idx = randperm(size(tmp,1),30);
            % tmp = tmp(idx,:);

            %normalize trial length to that of non waves
            tmp1 = stats_cl_hg(i).hg_nonwave;
            tmp1 = cell2mat(tmp1');

            for j=1:size(tmp1,2)
                tmp1(:,j) =smooth(tmp1(:,j),10);
            end


            if size(tmp,1) > size(tmp1,1)
                idx = randperm(size(tmp,1),size(tmp1,1));
                %idx = 1:size(tmp1,1);
                tmp = tmp(idx,:);
            end



            % remove bad channels
            tmp(:,bad_ch) = 1e-4*randn(size(tmp,1),length(bad_ch));


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


end

