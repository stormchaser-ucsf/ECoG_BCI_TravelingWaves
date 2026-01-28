function [stats,stats_hg] = planar_waves_stats(files,d2,...
    hilbert_flag,ecog_grid,grid_layout,elecmatrix,bpFilt,d1,cl_chk)

good_ch=ones(256,1);
good_ch([108 113 118])=0;
good_ch = logical(good_ch);
vec_field={};
stats={};kk=1;
stats_hg={};
for ii=1:length(files)
    disp(['Processing file ' num2str(ii) ' of ' num2str(length(files))])
    loaded=1;
    try
        load(files{ii})
    catch
        loaded=0;
        disp(['not loaded file  ' files{ii}])
    end

    if loaded==1

        kinax1 = find(TrialData.TaskState==1);
        kinax2 = find(TrialData.TaskState==2);
        kinax3 = find(TrialData.TaskState==3);
        kinax4 = find(TrialData.TaskState==4);

        %tmp=cell2mat(TrialData.BroadbandData);

        data2 = cell2mat(TrialData.BroadbandData(kinax2)');
        l2 =  length(data2);
        data4 = cell2mat(TrialData.BroadbandData(kinax4)');
        l4 = length(data4);
        data3 = cell2mat(TrialData.BroadbandData(kinax3)');
        l3 = length(data3);
        data1 = cell2mat(TrialData.BroadbandData(kinax1)');
        l1 = length(data1);

        data = [data1;data2;data3;data4];
        data_main=data;
        tmain = 1:size(data,1); % in ms the true time

        % constructing the output vector        
        output = NaN(size(data,1),1);
        if cl_chk==1
            t3start = l1+l2+1;
            t3end = t3start + l3;
            k=1;
            for t=t3start:200:(t3end-200)
                if k>length(TrialData.ClickerState)
                    break
                end
                output(t:t+199) = TrialData.ClickerState(k);
                k=k+1;
            end
        end


        %data = [data1;data2;data3];
        ds_fac=1e3/d2.SampleRate;
        l22=floor(l2/ds_fac); % length of the down sampled signal
        l11=floor(l1/ds_fac); % length of the down sampled signal

        % get the hG envelope
        % hg = filtfilt(bpFilt,data);
        % hg = abs(hilbert(hg));
        
        % get hg through filter bank approach
        Params=TrialData.Params;
        filtered_data=[];
        for k=9:16
            tmp = filtfilt(Params.FilterBank(k).b, ...
                Params.FilterBank(k).a, ...
                data);
            tmp=abs(hilbert(tmp));
            filtered_data = cat(3,filtered_data,tmp);
        end
        hg = squeeze(mean(filtered_data,3));

        % downsample to 50Hz
        hg = resample(hg,d2.SampleRate,1e3);

        % smooth it
        hg_smooth=[];
        for j=1:size(hg,2)
            hg_smooth(:,j) = smooth(hg(:,j),10);
        end      
        hg=hg_smooth;
        
        % get the mu signal phase of hG
        hg_mu = filtfilt(d2,hg);
        hg_mu = (hilbert(hg_mu));     

        % filter in mu band to get mu signal
        data = resample(data,d2.SampleRate,1000);
        df = filtfilt(d2,data);

        %downsample output vector to 50hz
        output = output(1:20:end);

        % get the hilbert transform of the signal
        if hilbert_flag
            df= hilbert(df);
        end

        % remove non-task periods
        df = df(l11+1:end-40,:);%remove last 800ms for b1,b6, last 1000ms for b3 hand 
        hg = hg(l11+1:end-40,:);%remove last 800ms for b1,b6, last 1000ms for b3 hand
        hg_mu = hg_mu(l11+1:end-40,:);
        output = output(l11+1:end-40);
        output(isnan(output)) = 1e-6;
        output(output~=TrialData.TargetID)=0;
        output(output==TrialData.TargetID)=1;

        % keep track of time
        tcut = tmain(l1:end-800); % what is being taken
        tcut = tcut(1:20:end);% down sampled to 50Hz        

        % detect planar waves across mini-grid location
        planar_val_time=[];planar_val_time_hg=[];
        planar_val_time_local=[];
        parfor t=1:size(df,1)
            %disp(t)
            
            % estimate planar waves across mini grid
            tmp = df(t,:);
            xph = tmp(ecog_grid);
                       
            planar_val = planar_stats_muller(xph);        
            planar_val_time(t,:,:) = planar_val;

            % doing it local over M1                        
            planar_val_time_local(t,:,:) = planar_val(3:8,1:5);           

            % wave detection for hg mu signal
            % tmp = hg_mu(t,:);
            % xph = tmp(ecog_grid);
            % planar_val = planar_stats_muller(xph);
            % planar_val_time_hg(t,:,:) = planar_val;
        end

        %%%% if performing local circular linear correlation around entire grid
        stab=[];stab_hg=[];stab_local=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));

            xt = planar_val_time_local(k,:,:);xt=xt(:);
            xtm1 = planar_val_time_local(k-1,:,:);xtm1=xtm1(:);
            stab_local(k-1) = - mean(abs(xt - xtm1));

            % xt = planar_val_time_hg(k,:,:);xt=xt(:);
            % xtm1 = planar_val_time_hg(k-1,:,:);xtm1=xtm1(:);
            % stab_hg(k-1) = - mean(abs(xt - xtm1));
        end       

        % figure;plot(zscore(stab))
        % hline(0)
        % hold on
        % plot(zscore(stab_local))
        
        stats(kk).stab = stab;
        stats(kk).vec_field = planar_val_time;        
        stats(kk).target_id = TrialData.TargetID;

        %%%%% SAVE TRIAL PERFORMANCE
        if cl_chk==1
            click_state = TrialData.FilteredClickerState(TrialData.FilteredClickerState>0);
            if mode(click_state) == TrialData.TargetID
                stats(kk).accuracy=1;
            else
                stats(kk).accuracy=0;
            end
            stats(kk).output=output;
        else
            stats(kk).output=0;
            stats(kk).accuracy=NaN;
        end

        %%%%% STABILITY AND WAVE DETECTION 
        % look 300 after start of state 2
        stab1 = zscore(stab(15:end));
        [out,st,stp] = wave_stability_detect(stab1);
        st = st+14;
        stp = stp+14;


        %stab = zscore(stab(1:end)); % 100:end for B3 HAND
        %[out,st,stp] = wave_stability_detect(stab);
        %st=st+99;%for b3 hand
        %stp=stp+99;%for b3 hand

        %%%% PLV analyses AND DATA STORING AROUND WAVE/NON WAVE EPOCHS
        %%%% extract hg envelope and mu only around waves and when decoder
        %%%% acc is correct (new)
        tmp={};tmp_mu={};tmp_hg_mu={};tmp_wave_stab={};        
        for k=1:length(st)

            % this just stores wave segments across all time, not tied to
            % decoder output. 
            tmp{k} = hg(st(k):stp(k),good_ch);
            tmp_mu{k} = df(st(k):stp(k),good_ch);
            tmp_hg_mu{k} = hg_mu(st(k):stp(k),good_ch);
            tmp_wave_stab{k} = stab(st(k):stp(k));

            % this part of code is to store wave segments when decoder
            % output was accurate. 
            % tmpp = output(st(k):stp(k));            
            % if cl_chk==1
            %     if sum(tmpp)/length(tmpp)>=0.5
            %         tmp{k} = hg(st(k):stp(k),good_ch);
            %         tmp_mu{k} = df(st(k):stp(k),good_ch);
            %         tmp_hg_mu{k} = hg_mu(st(k):stp(k),good_ch);
            %         tmp_wave_stab{k} = stab(st(k):stp(k));
            %     else
            %         tmp{k} = [];
            %         tmp_mu{k} = [];
            %         tmp_hg_mu{k} = [];
            %         tmp_wave_stab{k} = [];
            %     end
            % else
            %     tmp{k} = hg(st(k):stp(k),good_ch);
            %     tmp_mu{k} = df(st(k):stp(k),good_ch);
            %     tmp_hg_mu{k} = hg_mu(st(k):stp(k),good_ch);
            %     tmp_wave_stab{k} = stab(st(k):stp(k));
            % end
        end
        stats_hg(kk).hg_wave = tmp;
        stats_hg(kk).mu_wave = tmp_mu;
        stats_hg(kk).hg_mu_wave = tmp_hg_mu;
        stats_hg(kk).target_id = TrialData.TargetID;
        stats_hg(kk).wave_stab = tmp_wave_stab;

        % perform PLV analyses 
        tmp_mu = angle(cell2mat(tmp_mu'));
        tmp_hg_mu = angle(cell2mat(tmp_hg_mu'));
        res_wave = (exp(1i .* (tmp_mu - tmp_hg_mu)));


        %%%% extract hg envelope around non wave regions
        % till the first start
        tmp={};tmp_mu={}; tmp_hg_mu={};tmp_wave_stab={};k=1;           
        if st(1)>15+3  %change 15 to 1 etc. depending on when wave detections start
            tmp = cat(2,tmp,hg(15:(st(k)-1),good_ch));            
            tmp_mu = cat(2,tmp_mu,df(15:(st(k)-1),good_ch));            
            tmp_hg_mu = cat(2,tmp_hg_mu,hg_mu(15:(st(k)-1),good_ch));  
            tmp_wave_stab = cat(2,tmp_wave_stab,stab(15:(st(k)-1)));
        end
        % everything in between
        for j=1:length(stp)-1
            tmp = cat(2,tmp,hg((stp(j)+1) : (st(j+1)-1), good_ch));            
            tmp_mu = cat(2,tmp_mu,df((stp(j)+1) : (st(j+1)-1), good_ch));            
            tmp_hg_mu = cat(2,tmp_hg_mu,hg_mu((stp(j)+1) : (st(j+1)-1), good_ch));   
            tmp_wave_stab = cat(2,tmp_wave_stab,stab((stp(j)+1) : (st(j+1)-1)));
        end
        % get the last bit
        if (3+stp(end)) < (size(df,1))
            tmp = cat(2,tmp, hg((stp(end)+1):end,good_ch));
            tmp_mu = cat(2,tmp_mu, df((stp(end)+1):end,good_ch));
            tmp_hg_mu = cat(2,tmp_hg_mu, hg_mu((stp(end)+1):end,good_ch));
            tmp_wave_stab = cat(2,tmp_wave_stab,stab((stp(end)+1):end));
        end
        stats_hg(kk).hg_nonwave = tmp;
        stats_hg(kk).mu_nonwave = tmp_mu;
        stats_hg(kk).hg_mu_nonwave = tmp_hg_mu;
        stats_hg(kk).nonwave_stab = tmp_wave_stab;

        % perform PLV analyses 
        tmp_mu = angle(cell2mat(tmp_mu'));
        tmp_hg_mu = angle(cell2mat(tmp_hg_mu'));
        res_nonwave = (exp(1i .* (tmp_mu - tmp_hg_mu)));

        % contrast
        %figure;boxplot([abs(res_wave)' abs(res_nonwave)'])
        stats_hg(kk).plv_nonwave = res_nonwave;
        stats_hg(kk).plv_wave = res_wave;

        kk=kk+1;
    end  

end


end
% 