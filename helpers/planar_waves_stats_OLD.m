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

        % get the hG signal downsampled to 200Hz
        % data_hg = filtfilt(d3,data);
        % data_hg = abs(hilbert(data_hg));
        % data_hg = resample(data_hg,200,1000);        

        % resample and filter in mu band to get mu signal
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
        parfor t=1:size(df,1)
            %disp(t)
            
            % estimate planar waves across mini grid
            tmp = df(t,:);
            xph = tmp(ecog_grid);

            % smooth phase new
            % M = real(xph);
            % N = imag(xph);
            % tmp = smoothn({M,N},'robust');
            % M = tmp{1}; N = tmp{2};
            % xph = M + 1j*N;            
            
                     

            %%% estimate planar waves across the entire grid, fitting wave
            %%% patterns at a local subcluster around each electrode
            %planar_val = planar_stats_full(xph,grid_layout,elecmatrix);            
            %planar_val_time{t} = planar_val;
            
            planar_val = planar_stats_muller(xph);

            % smoothing
            % M = real(planar_val);
            % N = imag(planar_val);
            % tmp = smoothn({M,N},'robust');
            % M = tmp{1}; N = tmp{2};
            % planar_val = M+1j*N;


            planar_val_time(t,:,:) = planar_val;

            % wave detection for hg mu signal
            % tmp = hg_mu(t,:);
            % xph = tmp(ecog_grid);
            % planar_val = planar_stats_muller(xph);
            % planar_val_time_hg(t,:,:) = planar_val;
        end

        %%%% if performing local circular linear correlation around entire grid
        stab=[];stab_hg=[];
        for k=2:size(planar_val_time,1)
            xt = planar_val_time(k,:,:);xt=xt(:);
            xtm1 = planar_val_time(k-1,:,:);xtm1=xtm1(:);
            stab(k-1) = - mean(abs(xt - xtm1));

            % xt = planar_val_time_hg(k,:,:);xt=xt(:);
            % xtm1 = planar_val_time_hg(k-1,:,:);xtm1=xtm1(:);
            % stab_hg(k-1) = - mean(abs(xt - xtm1));
        end       
        
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

        % based on the time, you can actually go back and get the neural
        % features from TrialData.SmoothedNeuralFeatures in the high gamma
        % range. Can do it for wave epochs as well as non wave epochs. (All
        % the ones not selected for wave epochs will be characterized as
        % non wave epochs).  %st*20+1000 (state 1 is 1000)


        

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

        % % get hg features from smoothed neural data
        % hg_feat_wave={};
        % hg_feat_nonwave={};
        % tmp=TrialData.SmoothedNeuralFeatures;
        % wav_idx=[];
        % for k=1:length(st)
        %     st1 = st(k)*20+1000;
        %     stp1 = stp(k)*20+1000;
        %     st_bin = floor(st1/200);
        %     stp_bin = ceil(stp1/200);
        %     tmp_data = cell2mat(tmp(st_bin:stp_bin));
        %     tmp_data = tmp_data(1537:end,:);
        %     tmp_data = tmp_data(good_ch,:);
        %     hg_feat_wave{k}=tmp_data;
        %     wav_idx=[wav_idx st_bin:stp_bin];
        % end
        % hg_feat_wave = cell2mat(hg_feat_wave);


        % figure;plot(stab)
        % hline(0)
        % vline(st,'g')
        % vline(stp,'r')
        % hold on
        % plot(output,'--k','LineWidth',2)

        %%% Phase phase coupling between mu and high gamma only around
        %%% waves
        %get hG 
        % data_hg=filtfilt(bpFilt,data_main);        
        % ph_hg = angle(hilbert(data_hg));
        % % get mu 
        % data_mu=filtfilt(d1,data_main);     
        % ph_mu = angle(hilbert(data_mu));
        % x={};y={};
        % for k=1:length(st)
        %     tst = tcut(st(k));
        %     tstp = tcut(stp(k));
        %     [aa bb] = min(abs(tmain-tst));
        %     tst = tmain(bb);
        %     [aa bb] = min(abs(tmain-tstp));
        %     tstp = tmain(bb);
        %     %X = data_hg(tst:tstp,good_ch);
        %     %X = zscore(X);
        %     x{k} = ph_hg(tst:tstp,good_ch);
        %     y{k} = ph_mu(tst:tstp,good_ch);
        % end
        % 
        % % computing phase phase coupling
        % x=cell2mat(x'); % hg
        % y=cell2mat(y'); % mu 
        % % multiply mu by factor 12
        % sc=11;
        % y1 = wrapToPi(sc*y);
        % ppc_wave = (exp(1i .* (y1 -x)));
        % 
        % %%%%% phase phase coupling around non wave regions
        % x={};y={}; % for phase phase coupling
        % % till the first start        
        % if st>1
        %     tst = tcut(1);
        %     tstp = tcut(st(1))-1;
        %     [aa bb] = min(abs(tmain-tst));
        %     tst = tmain(bb);
        %     [aa bb] = min(abs(tmain-tstp));
        %     tstp = tmain(bb);
        % 
        %     x=cat(2,x,ph_hg(tst:tstp,good_ch));
        %     y=cat(2,y,ph_mu(tst:tstp,good_ch));
        % end
        % 
        % % everything in between 
        % for j=1:length(stp)-1
        %     tst = tcut(stp(j))+1;
        %     tstp = tcut(st(j+1))-1;
        %     [aa bb] = min(abs(tmain-tst));
        %     tst = tmain(bb);
        %     [aa bb] = min(abs(tmain-tstp));
        %     tstp = tmain(bb);
        %     x=cat(2,x,ph_hg(tst:tstp,good_ch));
        %     y=cat(2,y,ph_mu(tst:tstp,good_ch));
        % end
        % 
        % % get the last bit
        % if stp(end) < size(df,1)
        %     tst = tcut(stp(end))+1;
        %     tstp = tcut(end);
        %     [aa bb] = min(abs(tmain-tst));
        %     tst = tmain(bb);
        %     [aa bb] = min(abs(tmain-tstp));
        %     tstp = tmain(bb);
        %     x=cat(2,x,ph_hg(tst:tstp,good_ch));
        %     y=cat(2,y,ph_mu(tst:tstp,good_ch));
        % end
        % 
        % % computing phase phase coupling
        % x=cell2mat(x'); % hg
        % y=cell2mat(y'); % mu 
        % % multiply mu by factor 12
        % sc=11;
        % y1 = wrapToPi(sc*y);
        % ppc_nonwave = (exp(1i .* (y1 -x)));
        % 
        % % store
        % stats_hg(kk).ppc_wave = ppc_wave;
        % stats_hg(kk).ppc_nonwave = ppc_nonwave;


        %%%% PLV analyses
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
        tmp={};tmp_mu={}; tmp_hg_mu={};k=1;           
        % if st>1  %DONT NEED THIS AS WE START ONLY AT 300MS
        %     tmp = cat(2,tmp,hg(1:(st(k)-1),good_ch));            
        %     tmp_mu = cat(2,tmp_mu,df(1:(st(k)-1),good_ch));            
        %     tmp_hg_mu = cat(2,tmp_hg_mu,hg_mu(1:(st(k)-1),good_ch));  
        % end
        % everything in between
        for j=1:length(stp)-1
            tmp = cat(2,tmp,hg((stp(j)+1) : (st(j+1)-1), good_ch));            
            tmp_mu = cat(2,tmp_mu,df((stp(j)+1) : (st(j+1)-1), good_ch));            
            tmp_hg_mu = cat(2,tmp_hg_mu,hg_mu((stp(j)+1) : (st(j+1)-1), good_ch));            
        end
        % get the last bit
        if stp(end) < size(df,1)
            tmp = cat(2,tmp, hg((stp(end)+1):end,good_ch));
            tmp_mu = cat(2,tmp_mu, df((stp(end)+1):end,good_ch));
            tmp_hg_mu = cat(2,tmp_hg_mu, hg_mu((stp(end)+1):end,good_ch));
        end
        stats_hg(kk).hg_nonwave = tmp;
        stats_hg(kk).mu_nonwave = tmp_mu;
        stats_hg(kk).hg_mu_nonwave = tmp_hg_mu;

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

    %%%%% if just using one grid
    % b=cell2mat(planar_val_time');
    % r1=[b(1:end).rho];    
    % alp=[b(1:end).alp];    
    % stats(ii).corr = r1;    
    % stab=[];
    % for k=2:length(r1)
    %     tmp = cosd(alp(k)) + 1i*sind(alp(k));
    %     tmpm1 = cosd(alp(k-1)) + 1i*sind(alp(k-1));
    %     stab(k) = abs((r1(k)*tmp) - (r1(k-1)*tmpm1));
    % end
    %figure;plot(zscore(stab))

   

end


end
% 
% % plotting raw data
% M = real(xph);
% N = imag(xph);
% tmp = smoothn({M,N},'robust');
% M = tmp{1}; N = tmp{2};
% [XX,YY] = meshgrid( 1:size(xph,2), 1:size(xph,1) );
% figure;
% quiver(XX,YY,M,N);axis tight
% xphs = M + 1j*N;
% figure;
% imagesc(cos(angle(xphs)))
% 
% % get the vector field per Jacobs method
% [planar_val,planar_reg] = planar_stats_full(xph,grid_layout,elecmatrix);
% M = real(planar_val);
% N = imag(planar_val);
% tmp = smoothn({M,N},'robust');
% M = tmp{1}; N = tmp{2};
% [XX,YY] = meshgrid( 1:size(planar_reg,2), 1:size(planar_reg,1) );
% figure;
% quiver(XX,YY,M,N);axis tight
% 
% % comparing to gradient vector field muller method
% %M = real(xph);
% %N = imag(xph);
% %tmp = smoothn({M,N},'robust');
% %M = tmp{1}; N = tmp{2};
% %xphs = M + 1j*N;
% %xphs=xph;
% [pm,pd,dx,dy] = phase_gradient_complex_multiplication_NN( xphs, ...
%     1,-1);
% [XX,YY] = meshgrid( 1:size(xphs,2), 1:size(xphs,1) );
% ph=pd;
% M =  pm.*cos(ph);
% N =  pm.*sin(ph);
% tmp = smoothn({M,N},'robust');
% M = tmp{1}; N = tmp{2};
% figure;
% quiver(XX,YY,M,N);axis tight
% 
% % 
% % 
% 
% % % 
% % % % plotting the wave segment as arrows
% stab=zscore(stab);
% tt=(1:length(stab))*(1e3/50);
% figure;plot(tt,stab)
% [out,st,stp] = wave_stability_detect(stab,0);
% vline(tt(st),'g')
% vline(tt(stp),'r')
% h=hline(0);
% h.LineWidth=2;
% h.Color = 'k';
% 
% figure;
% v = VideoWriter('wave2_grad.avi');
% v.FrameRate = 6;
% open(v)
% for tt=62:76
%     planar_val = squeeze(planar_val_time(tt,:,:));
%     M = real(planar_val);
%     N = imag(planar_val);
%     tmp = smoothn({M,N},'robust');
%     M = tmp{1}; N = tmp{2};
% 
%     [XX,YY] = meshgrid( 1:size(planar_val,2), 1:size(planar_val,1) );
%     %figure;
%     quiver(XX,YY,M,N);axis tight
%     %title(num2str(tt))
%     drawnow
%     set(gca,'Ydir','reverse')
%     frame = getframe(gcf);
%     writeVideo(v,frame);
% end
% close(v)

% % making movie
% %tmp = df(9:32,:);
% tmp = df(99:111,:);
% tmp1=[];
% for i=1:size(tmp,2)
%     tmp1(:,i) = resample(tmp(:,i),1000,50);
% end
% v = VideoWriter('wave3.avi');
% v.FrameRate = 30;
% figure;
% open(v)
% for tt=1:size(tmp1,1)
% 
%     tmp = tmp1(tt,:);
%     xph = tmp(ecog_grid);
%     %figure;
%     imagesc(cos(angle(xph)))
%     %imagesc((real(xph)))
%     shading interp
%     title(['Time ' num2str(tt) 'ms'])
%     clim([-1 1])
%     colorbar 
%     drawnow
%     frame = getframe(gcf);
%     writeVideo(v,frame);
% end
% close(v)
% 
% % plotting duty cycle stats
% cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3')
% load('B3_waves_hand_stability_Muller.mat')
% 
% cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker')
% load('B1_waves_stability_Muller.mat')
% 
% cd('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6')
% load('B6_waves_stability_Muller.mat')
% 
