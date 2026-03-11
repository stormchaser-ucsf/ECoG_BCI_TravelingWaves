

% looking at PAC for EC189, thumb movement
% DMPAC - does high gamma power nest to the phase of LFOs? 
% look for only within the 'significant' channels
% controls: Have to take into account time perdiods when EC189 is not
% performing the movement. 

% DMPAC on the aysynch data first.

% PULL OUT EPOCHS FIRST. LOCKED TO MOVEMENT GO CUE, DOES HG ACTIVITY
% ELLICIT SIG LFO DMPAC AS COMPARED TO AVERAGE POWER IN REST PERIOD? 

%% loading data

clc;clear;

%cd('E:\DATA\ecog data\ECoG LeapMotion\Raw Data\EC189_ProcessingForNikhilesh')
%load('E:\DATA\ecog data\ECoG LeapMotion\Raw Data\\EC189_ProcessingForNikhilesh\EC189\ecog_data\proc_data.mat')
%load('E:\DATA\ecog data\ECoG LeapMotion\Raw Data\EC189_ProcessingForNikhilesh\EC189\ecog_data\B8_thumb_synch_proc_data')

cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh')
load('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189/ecog_data/proc_data.mat')

load('/media/user/Data/ecog_data/ECoG LeapMotion/Results/subj_data_25Hz_withKin_LMP.mat');

addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))

%% extracting the kinematic and neural data with markers

% kinematic time course
kin_time=kin_time{1}; % this in ecog time

% converting to hand basis
i=1;
kindata=[];
palm_pos = hand{i}.palm_pos;
basis = hand{i}.palm_basis;

for j=1:size(basis,1)
    palm_pos(j,:) = palm_pos(j,:) * squeeze(basis(j,:,:));
end

temp=[]; % temporary variable when extracting smoothed position data
for j=1:5 % fingers
    for k=1:5 % bones
        temp1 = fingers{i}(j).joints(k).pos;
        temp2=[];
        for l=1:size(temp1,1)
            temp2(l,:) =  (squeeze(basis(l,:,:))' * temp1(l,:)')';
            temp2(l,:)=temp2(l,:) - palm_pos(l,:);
        end
        temp=[temp temp2 ];
    end
end

temp1=mean(temp);
temp1=repmat(temp1,size(temp,1),1);
kindata = [kindata; (temp)-temp1];

offset = 50 * (1:size(kindata,2));
offset=repmat(offset,size(kindata,1),1);
temp=kindata+offset;
figure;plot(kin_time,temp)
axis tight

% plotting trial start, end etc. 
for i=1:length(trial_timings)
     vline(trial_timings(i).movement.cue(1).time,'r')
     vline(trial_timings(i).movement.cue(2).time,'y')
     vline(trial_timings(i).movement.cue(3).time,'g')    
end


figure;plot(kin_time,kindata(:,13))
hold on
for i=1:length(trial_timings)
    %if ~sum(i==[6 7 23])
     vline(trial_timings(i).movement.cue(1).time,'r') %rest
     vline(trial_timings(i).movement.cue(2).time,'y') % get ready
     vline(trial_timings(i).movement.cue(3).time,'g') % go   
    %end
end
ylim([-100 100])


kindata_full_length = kindata;

% first 286 channels are the neural channels 
% look at power spectrum 

%% synchronizing neural and kinematic data to 25Hz

% movement duration was set to 3s
% extract -500ms to 3s after green go cue for a total 3500ms per epoch

% upsample the kinematic data to the neural data range
[kindata_full_length_resampled ,kin_time_resample] = ...
    resample(kindata_full_length,kin_time,Fs);

% column channels
lfp=lfp';

% % find the bad-channels
bad_chI = subj_data(1).bad_chI_New;
bad_ch_idx = find(bad_chI==0);
% 
% figure;
% for i=1:length(bad_ch_idx)
%     tmp = lfp(:,bad_ch_idx(i));
%     [Pxx,F]=pwelch(tmp,[],[],[],Fs);
%     subplot(2,1,1)
%     plot(tmp);
%     axis tight
%     subplot(2,1,2)
%     plot(F,log10(abs(Pxx)));
%     axis tight
%     xlim([0 60])
%     sgtitle(num2str(bad_ch_idx(i)))
%     waitforbuttonpress
% end

% remove line noise

line_freq = 60;
harmonics = line_freq:line_freq:(Fs/2 - 1);   % all harmonics below Nyquist

bw_hz = 2;   % total notch width in Hz; try 1-3 Hz typically

for f0 = harmonics
    wo = f0/(Fs/2);          % normalized center frequency
    bw = bw_hz/(Fs/2);       % normalized bandwidth
    [b,a] = iirnotch(wo, bw);
    lfp = filtfilt(b, a, lfp);   % zero-phase filtering
end


% remove bad channels 
lfp=lfp(:,1:256);
lfp = lfp(:,bad_chI);

% median reference
lfp = lfp - median(lfp,2);




%% comparing power spectrum in move vs. rest periods
% sequence is rest, ready, go
ch= [204:209 219:224 187:193];
%ch= [219:224  187:193];
load('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/20251203/Robot3DArrow/133334/Imagined/Data0001.mat')
Params=TrialData.Params;
filterbank=[];k=1;
for i=9:16
    [b,a]=butter(3,Params.FilterBank(i).fpass/(Fs/2));
    filterbank(k).b=b;
    filterbank(k).a=a;
    filterbank(k).fpass=Params.FilterBank(i).fpass;
    k=k+1;
end

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',13,'HalfPowerFrequency2',15, ...
    'SampleRate',Fs);

bpFilt1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',13,'HalfPowerFrequency2',19, ...
    'SampleRate',Fs);


% get mu and hg
mu = [];hg=[];
alp=[];
mu = filtfilt(bpFilt,lfp(:,ch));
mu = abs(hilbert(mu));
mu = mean(mu,2);
mu=zscore(mu);

alp = filtfilt(bpFilt1,lfp(:,ch));
alp = abs(hilbert(alp));
alp = mean(alp,2);
alp=zscore(alp);

tmp_hg=[];
for j=1:8
    tmp = filtfilt(filterbank(j).b,filterbank(j).a,...
        lfp(:,ch));
    tmp = abs(hilbert(tmp));
    tmp_hg = cat(3,tmp_hg,tmp);
end
tmp_hg = squeeze(mean(tmp_hg,3));
hg = squeeze(mean(tmp_hg,2));
hg=zscore(hg);

% plot 
% also extract power spectrum within each signal portion (1/f)
% figure;
% hold on
% plot(lfp_time,smooth(zscore(hg),100))
% plot(lfp_time,smooth(zscore(mu),100))
mu_ep=[];
hg_ep=[];
alp_ep=[];
kin_ep=[];
osc_clus=[];
stats=[];
pow_freq=[];
ffreq=[];
for i=1:length(trial_timings)
     timings = [trial_timings(i).movement.cue.time];
     % for j=1:length(timings)
     %     vline(timings(1),'r')
     %     vline(timings(2),'y')
     %     vline(timings(3),'g')
     % end
     go_time = trial_timings(i).movement.cue(2).time; %anin
     st = go_time-1;
     stp = go_time+8;
     
     index = (kin_time_resample >= st) .* (kin_time_resample<=stp);
     kin_ep(:,i) = kindata_full_length_resampled(logical(index),13);
     
     index = (lfp_time >= st) .* (lfp_time<=stp);
     mu_ep(:,i) = mu(logical(index));
     hg_ep(:,i) = hg(logical(index));
     alp_ep(:,i) = alp(logical(index));


     st = trial_timings(i).movement.cue(2).time; %anin
     stp = trial_timings(i).movement.cue(3).time; %anin
     %stp=st+4;
     index = (lfp_time >= st) .* (lfp_time<=stp);
     data = lfp(logical(index),:);
     
     spectral_peaks=[];
     stats_tmp=[];
     parfor ii=1:size(data,2)
         x = data(:,ii);
         if length(x)>=1024
             [Pxx,F] = pwelch(x,1024,512,1024,1e3);
             %[Pxx,F] = pwelch(x,hamming(1000),500,2048,1000);
             pow_freq = [pow_freq;Pxx' ];
             ffreq = [ffreq ;F'];
             idx = logical((F>0) .* (F<=40));
             F1=F(idx);
             F1=log2(F1);
             power_spect = Pxx(idx);
             power_spect = log2(power_spect);
             tb=fitlm(F1,power_spect,'RobustOpts','on');
             stats_tmp = [stats_tmp tb.Coefficients.pValue(2)];
             bhat = tb.Coefficients.Estimate;
             x = [ones(length(F1),1) F1];
             yhat = x*bhat;

             %plot
             % figure;
             % plot(F1,power_spect,'LineWidth',1);
             % hold on
             % plot(F1,yhat,'LineWidth',1);

             % get peaks in power spectrum at specific frequencies
             power_spect = zscore(power_spect - yhat);
             [aa ,bb]=findpeaks(power_spect);
             peak_loc = bb(find(power_spect(bb)>1));
             freqs = 2.^F1(peak_loc);
             pow = power_spect(peak_loc);

             %store
             spectral_peaks(ii).freqs = freqs;
             spectral_peaks(ii).pow = pow;
         end
     end


     % getting oscillation clusters
     if ~isempty(spectral_peaks)
         osc_clus_tmp=[];
         for f=2:40
             ff = [f-1 f+1];
             tmp=0;ch_tmp=[];
             for j=1:length(spectral_peaks)
                 freqs = spectral_peaks(j).freqs;
                 for k=1:length(freqs)
                     if ff(1) <= freqs(k)  && freqs(k) <= ff(2)
                         tmp=tmp+1;
                         ch_tmp = [ch_tmp j];
                     end
                 end
             end
             osc_clus_tmp = [osc_clus_tmp tmp];
         end
         osc_clus = [osc_clus;osc_clus_tmp];
     end

end


% plot oscillation clusters
f=2:40;
figure;
hold on
plot(f,osc_clus/227,'Color',[.5 .5 .5 .5],'LineWidth',.5)
plot(f,mean(osc_clus,1)/227,'b','LineWidth',2)
xlabel('Freq.')
ylabel('Prop of channels')
plot_beautify
title('1/f sig.')

% plot power spectrum
figure;
hold on
plot(ffreq(1,:),log10(pow_freq'),'Color',[.5 .5 .5 .5])
plot(ffreq(1,:),log10(mean(pow_freq,1)),'k','LineWidth',2)
xlim([0 200])


m = mean(mu_ep(1:254*2,:),1);
s = std(mu_ep(1:254*2,:),1);
mu_ep = (mu_ep-m)./s;

m = mean(hg_ep(1:254*2,:),1);
s = std(hg_ep(1:254*2,:),1);
hg_ep = (hg_ep-m)./s;

m = mean(alp_ep(1:254*2,:),1);
s = std(alp_ep(1:254*2,:),1);
alp_ep = (alp_ep-m)./s;

figure;
hold on
tt=-1:(1/Fs):8;
if length(tt)>size(alp_ep,1)
    tt=tt(1:end-1);
end
plot(tt,mean(hg_ep,2),'b','LineWidth',1)
plot(tt,mean(mu_ep,2),'r','LineWidth',1)
plot(tt,mean(alp_ep,2),'k','LineWidth',1)
vline([0 4])
hline(0)
xlim([-0.5 5])
ylim([-3 6])
legend({'hG','narrow beta','beta'})
xlabel('Time (s)')
ylabel('Z score')
plot_beautify
axis tight

a=alp_ep(500:1000,:);
b=mu_ep(500:1000,:);
a=median(a,1);
b=median(b,1);[p,h]=signrank(a,b)

% so kinda the same concept, except the frequency moved up to a higher one

%%
kindata1=[];ecog=[];
%bad_trials = [5 7 10 11 17 21 23 24 27]; % middle
bad_trials = [1 12 21 22];
for i=1:length(trial_timings)-1
    if sum(i==bad_trials) == 0
        go_time = trial_timings(i).movement.cue(3).time; %anin
        st = go_time-0.5;
        stp = go_time+3;
        index = (kin_time_resample >= st) .* (kin_time_resample<=stp);
        kindata1 = [kindata1; kindata_full_length_resampled(logical(index),:)];
        
        index = (lfp_time >= st) .* (lfp_time<=stp);
        ecog=[ecog;lfp(:,logical(index))'];
    end
end
% 


bad_ch = [1 2 8 12 22 23 65 68 71 77 106 118 123 124 127 128 129 130 160 172 ...
    191 193 250 254 262 263 275 283 286 ];

bad_chI = ones(286,1);
bad_chI(bad_ch)=0;
bad_chI = logical(bad_chI);

% z-score
ecog=zscore(ecog);
ecog = ecog(:,1:286);

% median reference the signal first 286 channels
median_ref  = median(ecog(:,bad_chI),2);
ecog = ecog-median_ref;

% LFO envelope
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
         'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
         'SampleRate',Fs);
fvtool(bpFilt)
lfo_data=[];
for j=1:size(ecog,2)    
    temp=[std(ecog(:,j))*(randn(4000,1)+mean(ecog(:,j))); ecog(:,j); ...
        std(ecog(:,j))*(randn(4000,1)+mean(ecog(:,j)))];
    temp=filtfilt(bpFilt,temp);
    temp1=temp;
    temp=abs(hilbert(temp));
    temp=temp(4001:end-4000);
    temp1=temp1(4001:end-4000);
    %kindata(:,j) = filtfilt(d2,kindata(:,j));
    ecog(:,j) = temp;
    lfo_data(:,j) = temp1;
end

% downsample both kin and ecog data to 25Hz.
time = (0:size(ecog,1)-1) * (1/Fs);
ecog = resample(ecog,time,25);
lfo_data = resample(lfo_data,time,25);
kindata = resample(kindata1,time,25);

lfp=lfp';

cd('E:\DATA\ecog data\ECoG LeapMotion\EC189_processed_data_asynch_full_hand')
save ecog_kinematics_asynch_pinky_NN_25Hz ecog kindata lfo_data lfp time Fs bad_ch bad_chI -v7.3


%% regular CCA processing steps for EC189
% aim is to get the 'sig' electrodes to DMPAC
% GO TO EC189_CCA CODE FOR ALL THE NECESSARY STEPS 
clear
load('E:\DATA\ecog data\ECoG LeapMotion\Raw Data\EC189_ProcessingForNikhilesh\EC189\ecog_data\proc_data.mat')
cd('E:\DATA\ecog data\ECoG LeapMotion\Results')
load('EC189_Thumb_Asynch.mat', 'Wa1p','bad_chI','bad_ch')


% loading the sig. channels from the AR model
load Sig_Wts_AR_Model_EC189

% get the results at the p=0.01 level, FDR corrected
% doing the stats
Wa=results.Wa;
Wa_boot=results.Wa_boot;
Wap=Wa;p_values=[];
for ii=1:size(Wa,2)
    temp=abs(Wa_boot(:,:,ii));
    temp1=abs(Wap(:,ii));
    p=[];
    for i=1:length(temp1)
        p(i) = sum(temp(:,i)>temp1(i))/length(temp(:,i));
    end
    p_values(:,ii)=p;
    [pfdr pval]=fdr(p,.05,'Parametric');
    %         if sum(p<=pfdr) == 0
    %             [pfdr pval]=fdr(p,.05,'Parametric');
    %         end
    p(p<=pfdr)=0;
    p(p~=0)=1;
    p=1-p;
    p=logical(p);
    Wap(:,ii)=Wap(:,ii).*p';
end

% mean p-values at sig. indiv. channels
Wa1p_pval=[];
for i=1:size(Wa,1)
    if sum(Wap(i,:)>0)
        temp_pval=p_values(i,find(Wap(i,:)~=0));
        temp_pval(temp_pval==0)=1/1001;
        Wa1p_pval(i) = mean(temp_pval);
    else
        Wa1p_pval(i) = 0;
    end
end

% taking into account bad channels
Wa1p=zeros(size(Wa1p));
Wa1p_pvalues=zeros(size(Wa1p,1),1);
k=1;
for i=1:size(Wa1p,1)
    if sum(bad_ch==i)==1
        Wa1p(i,:)= 0;
        Wa1p_pvalues(i)=0;
    else
        Wa1p(i,:)=Wap(k,:);
        Wa1p_pvalues(i) = Wa1p_pval(k);
        k=k+1;
    end
end

% make sure lfp matrix is column matrix
if size(lfp,2) > size(lfp,1)
    lfp=lfp';
end

% remove line noise and harmonics up to 4
freq = [60 120 180 240];
for i=1:length(freq)
    bsFilt = designfilt('bandstopiir','FilterOrder',6, ...
        'HalfPowerFrequency1',freq(i)-0.5,'HalfPowerFrequency2',freq(i)+0.5, ...
        'SampleRate',Fs);
    lfp=filtfilt(bsFilt,lfp);
end

% High gamma
hgFilt = designfilt('bandpassiir','FilterOrder',8, ...
         'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
         'SampleRate',Fs);
fvtool(hgFilt)
hg = filtfilt(hgFilt,lfp);
hg = abs(hilbert(hg));

%hg=zscore(hg);

% create HG epoochs during movement and rest period 
hg_mvmt={};hg_rest={};
for i=1:length(trial_timings)
    % go cue
    go_time = trial_timings(i).movement.cue(3).time; %anin
    st = go_time+0.1;
    stp = go_time+2.6;    
    index = (lfp_time >= st) .* (lfp_time<=stp);
    hg_mvmt{i}=hg(logical(index),:);
    
    % rest period: 500ms after rest till Get ready cue
    rest_time = trial_timings(i).movement.cue(1).time; %anin
    get_ready_time = trial_timings(i).movement.cue(2).time; 
    st = rest_time+0.1;
    stp = rest_time+2.6;
    %st = get_ready_time-2;
    %stp = get_ready_time;
    index = (lfp_time >= st) .* (lfp_time<=stp);
    hg_rest{i} = hg(logical(index),:);
end

% perform DMD analyses i.e. LFO oscillatory power of HG amplitude across
% the 'sig' channels
% during the rest and move periods
sig_ch = sum(Wa1p,2);
sig_ch = sig_ch~=0;
res_pow=[];mvmt_pow=[];
resp={};mvmtp={};smode={};
for i=1:length(hg_rest)
    disp(i)
    temp = hg_rest{i};
    temp=temp(:,1:length(sig_ch));
    X=temp(:,sig_ch);
    [P, f ]=dmd_alg(X',Fs,.1,4);
    index = (f>=0.1) .* (f<=4);
    %index = f<=4;
    res_pow = [res_pow mean(P(logical(index)))];
    resp(i).f = f;
    resp(i).P = P;
    
    temp = hg_mvmt{i};
    temp=temp(:,1:length(sig_ch));
    X=temp(:,sig_ch);
    [P, f,Phi,lambda,Xhat,~,Z,rf]=dmd_alg(X',Fs,.5,4);
    %index = f<=4;
    index = (f>=0.1) .* (f<=4.2);
    mvmt_pow = [mvmt_pow mean(P(logical(index)))];
    mvmtp(i).f=f;
    mvmtp(i).P=P;    
    smode(i).Phi = Phi;
    smode(i).f = f;
    smode(i).Xhat = Xhat;
    smode(i).lambda = lambda;
    smode(i).Z = Z;
    smode(i).rf=rf;
end


% average delta power 
avg_mvmt_pow=[];avg_res_pow=[];
for i=1:length(mvmtp)
    f=mvmtp(i).f;
    P=mvmtp(i).P;
    index = logical((f>=0.5) .* (f<=4));
    avg_mvmt_pow(i)=mean(P(index));    
    
    f=resp(i).f;
    P=resp(i).P;
    index = logical((f>=0.5) .* (f<=4));
    avg_res_pow(i)=mean(P(index));    
end


% NEW
% looking at average wts magnitude per 1Hz frequency bins 
bins = [0.6:0.5:4.1];
wts=[];
for i=1:length(bins)    
    f=bins(i)+[-0.5 0.5];
    temp_wts=[];    
    for j=1:length(smode)
        temp=smode(j).f;      
        index = logical( (temp>=f(1)) .* (temp<=f(2)) );
        temp_spatial = smode(j).Phi;
        temp_spatial = temp_spatial(:,index);
        temp_wts(j,:) = mean(temp_spatial,2);        
    end
    wts(:,i) = mean(temp_wts,1);
end

%%%% NEW NEW NEW %%%
% the bootstrap test on weight magnitudes: get the same number of trials as
% the original with circ-shifting time. compute mean. Do this a bunch of
% times say about 50 for now. 
% For circular shuffling: power spctra
% For spatial shuffling: weights magnitude 
tic
bins = [0.6:0.5:4.1]; 
bins1 = [1.1:1:50.1;]; 
wts_boot=[];
pow_boot=[];
pow_boot_delta=[];
boot_stats=[];
boot_type=1; % 1 - circular shuffling and 2 - spatial shuffling 
for iter=1:1000
    disp(iter)
    spatial_boot={};
    parfor i=1:length(hg_mvmt)
        temp = hg_mvmt{i};
        temp=temp(:,1:286);
        X=temp(:,sig_ch);
        if boot_type==1
            % circularly shuffling each channel's activity i.e. for the power
            % stats
            for j=1:size(X,2)
                X(:,j) = circshift(X(:,j),randperm(size(X,1),1));
            end
        end
        if boot_type==2
            % swap electrode position activity i.e. for the spatial stats
            for j=1:size(X,2)
                X(:,j) = circshift(X(:,j),randperm(size(X,1),1));
            end
            X = X(:,randperm(size(X,2)));
        end
        [P, f,Phi]=dmd_alg(X',Fs,.1,4);
        index = (f>=0.5) .* (f<=4);
        spatial_boot(i).Phi = Phi;
        spatial_boot(i).f = f;
        spatial_boot(i).P =P;
    end
    
    % spatial stats
    wtsb=[];
    for i=1:length(bins)
        f1=bins(i)+[-0.5 0.5];
        temp_wts=[];
        for j=1:length(spatial_boot)
            temp=spatial_boot(j).f;
            index = logical( (temp>=f1(1)) .* (temp<=f1(2)) );
            temp_spatial = spatial_boot(j).Phi;
            temp_spatial = temp_spatial(:,index);
            temp_wts(j,:) = mean(temp_spatial,2);
        end
        wtsb(:,i) =  mean(temp_wts,1);
    end
    wts_boot(iter,:,:) = wtsb;
    
    % power stats
    pow=[];
    for i=1:length(bins1)
        f1=bins1(i)+[-1 1];
        temp_pow=[];
        for j=1:length(spatial_boot)
            temp=spatial_boot(j).f;
            index = logical( (temp>=f1(1)) .* (temp<=f1(2)) );
            temp_pow(j,:) = mean(spatial_boot(j).P(index));
        end
        pow(:,i) = mean(temp_pow);
    end
    pow_boot(iter,:) = pow;
    
    % delta power stats
    temp=[];
    for i=1:length(spatial_boot)
        f=spatial_boot(i).f;
        P=spatial_boot(i).P;
        index = logical((f>=0.5) .* (f<=4));
        temp(i) = mean(P(index));      
    end
    pow_boot_delta(iter)=mean(temp);
    
    %storing the boot stats
    temp=[];
    temp.f=spatial_boot.f;
    temp.P=spatial_boot.P;
    boot_stats{iter} = temp;
end
toc
sum(pow_boot_delta>mean(avg_mvmt_pow))/length(pow_boot_delta)
figure;hist(pow_boot_delta);vline(mean(avg_mvmt_pow))

% stats on the spatial modes in 1Hz bins. Stats: Is the average weight
% magnitude per channel and freq bin > than what would be expected under
% the null hypothesis? 
wtsp=[];
for i=1:length(bins)   
    for j=1:size(wts,1)
       wts_null = squeeze(wts_boot(:,j,i));
       wtsp(j,i) =  sum( abs(wts(j,i)) <= abs(wts_null) )/length(wts_null);
    end
end

% plotting the power spectra comparisons between movement and the null
% distriubtions
freq = 1.1:1:50.1;
res_avg=[];
mvmt_avg=[];
boot_avg=[];
nontask_avg=[];
for i=1:length(freq)
    for j=1:length(resp)
        f = freq(i)+[-1 1];
        %f = freq(i)+[-0.5 0.5];
        % rest
        index = logical((resp(j).f >= f(1)) .* (resp(j).f <= f(2)));
        res_avg(j,i) = nanmean(resp(j).P(index));
        %mvmt
        index = logical((mvmtp(j).f >= f(1)) .* (mvmtp(j).f <= f(2)));
        mvmt_avg(j,i) = nanmean(mvmtp(j).P(index));  
    end   
%     for  k=1:length(nontaskp)
%         index = logical((boot(k).f >= f(1)) .* (boot(k).f <= f(2)));
%         nontask_avg(k,i) = nanmean(nontaskp(k).P(index));
%     end
end
pow_boot = sort(pow_boot,'ascend');
figure;hold on
plot(freq,mean(mvmt_avg,1))
plot(pow_boot(25,:),'--r')
plot(pow_boot(975,:),'--r')

% bar plots of avg. power within delta band
idx = logical((freq>=0.5) .* (freq<=5));
lower_ci = mean(pow_boot(25,idx));
upper_ci = mean(pow_boot(975,idx));
figure;hold on
scatter(ones(size(mvmt_pow))+0.1*randn(size(mvmt_pow)),mvmt_pow)
xlim([0 2])
hline(lower_ci)
hline(upper_ci)
plot(1,mean(mvmt_pow),'+r','MarkerSize',15)

% plotting the mean value vs. the bootstrapped distribution
idx = logical((freq>=0.5) .* (freq<=5));
temp=pow_boot(:,idx);
temp=mean(temp,2);
figure;hist(temp)
vline(mean(mvmt_pow));

% plotting brain map of important channels and the average LFO dynamics 
% things to plot are average 1.5Hz rhythm  say for example, with spatial
% mode and then also complete PC1 vs. PC2 for entire LFO space
lfo_dyan=[];fdyan=[];
for i=1:length(smode)
    temp = smode(i).rf;
    [aa bb]=(min(abs(temp-0.5)));    
    temp=smode(i).Z(bb,:);
    fdyan(i) = smode(i).rf(bb);
    temp=real(temp(:,1:1270));
    lfo_dyan(:,i) =zscore(mean(temp,1));
    %[coeff,score,latent]=pca(temp');    
end
% plotting the average dynamics with bootstrapped C.I.
t=(1/Fs)*(1:length(temp));
figure;plot(t,mean((lfo_dyan),2))
temp=mean((lfo_dyan),2);
% the frequencies extracted
figure;stem(fdyan)
% the power spectrum 
[Pxx,F] = pwelch(temp,length(temp),0,[],Fs);
figure;stem(F,Pxx);xlim([0 4])

% plotting the corresponding sig. spatial mode, with mag and phase
% plot the top n% of channels 
q=1;
ch = find(sig_ch==1);
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
w=wts(:,q);
% %%% by p-value
% p = wtsp(:,q);
% p(p>0.05)=0;
% p(p~=0)=1;
% w(p==0)=0;
%%% by ch-magnitude
[aa bb]=sort(abs(w),'descend');
p=zeros(length(w),1);
p(bb(1:ceil(length(w)*0.15)))=1; % top 15% of channels 
w(p==0)=0;
l=length(w);
ph=angle(w);
val = abs(w)/max(abs(w));
phMap = linspace(-pi,pi,l)';
ChColorMap=parula(round(l/2));
ChColorMap = [ChColorMap;flipud(ChColorMap)];
if rem(l,2)~=0
    ChColorMap = ChColorMap(1:end-1,:);
end
for j=1:l
    ms = val(j)*(15);
    [aa bb]=min(abs(ph(j) - phMap));
    c=ChColorMap(bb,:);
    if w(j)~=0
        e_h = el_add(elecmatrix(ch(j),:), 'color', c,'msize',ms);
    end
end
% colormap plotting
figure;imagesc(phMap')
colormap(ChColorMap)
axis off
set(gcf,'Color','w')

% making a movie using reconstructed dynamics 
% have to first create requisite dynamics
sig_ch = sum(Wa1p,2);
sig_ch = sig_ch~=0;
recon=[];
lo=0.5;up=4;
parfor i=1:length(hg_mvmt)
    disp(i)    
    temp = hg_mvmt{i};
    temp=temp(:,1:286);
    X=temp(:,sig_ch);
    [P, f,Phi,lambda,Xhat,~,Z,rf]=dmd_alg(X',Fs,lo,up);
    recon(i,:,:) = (Xhat);    
end
% get the mean
Xhat = squeeze(real(mean(recon,1)));
% get it only over the important electrodes
I = find(w~=0);
Xhat=Xhat(I,:);
% get the number of the sig. electrodes across the grid
ch_plot = ch(I);
% get frames
M={};figure
for i=1:size(Xhat,2)     
    disp(i)
    c_h = ctmr_gauss_plot(cortex,elecmatrix(ch_plot,:),...
        Xhat(:,i),'lh');
    view(-100,25)
    caxis([-1 1])
    set(gcf,'Color','w')
    set(gca,'FontSize',20)
    title(num2str(i));
    M{i}=getframe;  
    clf
end
% make a movie
vidObj = VideoWriter('EC189 hG Recon Dynamics 2.1Hz.avi');
vidObj.FrameRate = 100;
open(vidObj);
figure;
for i=1:size(M,2)
    imagesc(frame2im(M{i}));
    text(7.2,7.2,num2str(i));
    pause(0.001)
    writeVideo(vidObj,M{i});    
end
close(vidObj);


% making a movie using just the average hG dynamics and spatial modes
Xhat=[];
d = mean((lfo_dyan),2);
dp=angle(hilbert(d));
ph = ph(ph~=0);
% low pass filter
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
         'PassbandFrequency',10,'PassbandRipple',0.2, ...
         'SampleRate',Fs);
fvtool(lpFilt)
% channels are phase offset from each other 
l=[];
for i=1:length(ph)
    % find the temporal offset less than 1e-3
    offset=1;k=1;
    while offset>0.1
       offset = abs(ph(i)-dp(k));
       k=k+1;
    end
    k=k-1;
    l=[l k]; 
    temp=circshift(d,k);
    Xhat(i,:)=filtfilt(lpFilt,temp);
end
figure;plot(dp)
hline(ph)
vline(l,'c')
% make the movie
M={};figure
for i=1:size(Xhat,2)     
    disp(i)
    c_h = ctmr_gauss_plot(cortex,elecmatrix(ch_plot,:),...
        Xhat(:,i),'lh');
    view(-100,25)
    caxis([-0.4 0.4])
    set(gcf,'Color','w')
    set(gca,'FontSize',20)
    title(num2str(i));
    M{i}=getframe;  
    clf
end

clear lfp
%save EC189_DMPAC_Stats_Thumb_Asynch -v7.3
save EC189_DMPAC_Stats_Thumb_Asynch_AR -v7.3

% PCA
Xhat = squeeze(real(mean(recon,1)));
[coeff,score,latent]=pca(Xhat');
figure;stem(cumsum(latent)./sum(latent))
cmap = parula(length(score));
figure;hold on
for i=1:size(score,1)
    plot3(score(i,1),score(i,2),score(i,3),'.','Color',cmap(i,:))
end
grid on
% plot the coeff on the brain 
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
ch = find(sig_ch==1);
w=(coeff(:,3));
val = abs(w)/max(abs(w));
l=length(w);
ph=angle(w);
phMap = linspace(0,2*pi,l)';
ChColorMap=parula(round(l/2));
ChColorMap = [ChColorMap;flipud(ChColorMap)];
if rem(l,2)~=0
    ChColorMap = ChColorMap(1:end-1,:);
end
for j=1:l
    ms = val(j)*(15);
    [aa bb]=min(abs(ph(j) - phMap));
    c=ChColorMap(bb,:);
    if w(j)~=0
        e_h = el_add(elecmatrix(ch(j),:), 'color', 'b','msize',ms);
    end
end

% power spectra
for i=1:10
    [Pxx,F]=pwelch(score(:,i),[],[],[],Fs);
    figure;stem(F,abs(Pxx));xlim([0 10])
    title(num2str(i));
end

% PCA on all trials
Xhat=[];
for i=1:size(recon,1)
    Xhat = [Xhat squeeze(recon(i,:,:))];
end
[coeff,score,latent]=pca(real(Xhat'));
figure;stem(cumsum(latent)./sum(latent))
figure;plot(score(1:1272,1),score(1:1272,2),'.')
%figure;plot3(score(:,3),score(:,1),score(:,2),'.')
grid on
pc1=[];
pc2=[];
pc3=[];
pc5=[];
k=1;
for i=1:1272:length(score)
   pc1(k,:) = score(i:i+1272-1,1); 
   pc2(k,:) = score(i:i+1272-1,2);    
   pc3(k,:) = score(i:i+1272-1,3);    
   pc5(k,:) = score(i:i+1272-1,53);
   k=k+1;
end
temp_pc1=mean(pc1,1);
temp_pc2=mean(pc2,1);
temp_pc3=mean(pc3,1);
temp_pc5=mean(pc5,1);
cmap = parula(length(temp_pc1));
figure;hold on
for i=1:length(cmap)
   plot(temp_pc1(i),temp_pc2(i),'.','MarkerSize',10,'Color',cmap(i,:))     
end
axis tight
axis off
set(gcf,'Color','w')

figure;hold on
for i=1:length(cmap)
   plot3(temp_pc1(i),temp_pc2(i),temp_pc3(i),'.','MarkerSize',10,'Color',cmap(i,:))     
end
axis tight
axis off
set(gcf,'Color','w')
axis on
grid on



%
% % during the movement period
% total_pow=[];figure;hold on
% for i=1:length(mvmt_pow)
%     total_pow = [total_pow ;res_pow(i) mvmt_pow(i)];
%     plot(i,res_pow(i),'.r','MarkerSize',15)
%     plot(i+0.05,mvmt_pow(i),'.b','MarkerSize',15)
% end
% set(gcf,'Color','w')
% xlabel('Trials')
% ylabel('Power')
% set(gca,'FontSize',16)
% set(gca,'LineWidth',2)
% legend('Delta modes of HG during rest','Delta modes of HG during thumb flexion')
% 
% res=total_pow(:,2)-total_pow(:,1);
% [h p]=ttest(res)
% sum(res>0)
% 
% res = 100*((mvmt_pow - res_pow)./res_pow)

% plotting the average power spectrum in 2Hz bins 
freq = 1.1:1:50.1;
%freq = 0.5:1:50.5;
res_avg=[];
mvmt_avg=[];
boot_avg=[];
for i=1:length(freq)
    for j=1:length(resp)
        f = freq(i)+[-1 1];
        %f = freq(i)+[-0.5 0.5];
        % rest
        index = logical((resp(j).f >= f(1)) .* (resp(j).f <= f(2)));
        res_avg(j,i) = nanmean(resp(j).P(index));
        %mvmt
        index = logical((mvmtp(j).f >= f(1)) .* (mvmtp(j).f <= f(2)));
        mvmt_avg(j,i) = nanmean(mvmtp(j).P(index));  
    end
%     for k=1:length(boot)
%         %boot
%         index = logical((boot(k).f >= f(1)) .* (boot(k).f <= f(2)));
%         boot_avg(k,i) = nanmean(boot(k).P(index));          
%     end
end
% just plot the mean
figure;plot(freq,nanmean(res_avg,1))
hold on
plot(freq,nanmean(mvmt_avg,1))

r1 = mean(res_avg(:,1:4),2);
m1 = mean(mvmt_avg(:,1:4),2);
[ h p]=ttest(m1-r1)

% plot with shading of C.I.
% get bootstrap estimates of mean from the circ shuffling scheme 
boot_avg_boot=bootstrp(1000,@mean,boot_avg);
boot_avg_boot = sort(boot_avg_boot,'ascend');
figure;hold on
plot(freq,boot_avg_boot(25,:),'--r','LineWidth',1.5)
plot(freq,boot_avg_boot(975,:),'--r','LineWidth',1.5)
plot(freq,nanmean(mvmt_avg,1),'b','LineWidth',2)
axis tight
set(gca,'FontSize',16)
xlabel('Frequency')
ylabel('Power')
set(gcf,'Color','w')
set(gca,'LineWidth',2)
title('Power Spectra of Spatial hG dynamics')


% Bar plots with the C.I lines drawn and with individual data points
% corresponding to above data
idx = logical((freq>=0.5) .* (freq<=4));
lower_ci = mean(boot_avg_boot(25,idx));
upper_ci = mean(boot_avg_boot(975,idx));
figure;hold on
scatter(ones(size(mvmt_pow))+0.1*randn(size(mvmt_pow)),mvmt_pow)
xlim([0 2])
hline(lower_ci)
hline(upper_ci)
plot(1,mean(mvmt_pow),'+r','MarkerSize',15)


% Bar plots with C.I. derived from 95% distribution of circular stat boot
% values
boot_pow = sort(boot_pow,'ascend');
figure;
scatter(ones(size(mvmt_pow))+0.1*randn(size(mvmt_pow)),mvmt_pow)
xlim([0 2])
hline(boot_pow(3))
hline(boot_pow(96))


% 
% cd('E:\DATA\ecog data\ECoG LeapMotion\Results')
% clearvars -except total_pow res_pow mvmt_pow hg_mvmt hg_rest Wa1p Fs
% save EC189_Asynch_Thumb_DMDPAC_HG_Delta -v7.3


%% PLOTTING RESULTS OF DMPAC 

clear
cd('E:\DATA\ecog data\ECoG LeapMotion\Results')
load('EC189_Asynch_Thumb_DMDPAC_HG_Delta')
load('EC189_Thumb_Asynch','cortex','elecmatrix');

% perform DMD one time 
sig_ch = sum(Wa1p,2);
sig_ch = sig_ch~=0;
temp = hg_mvmt{18};
temp=temp(:,1:286);
X=temp(:,sig_ch);
[P, f ,Phi, lambda, Xhat, z0, Z]=dmd_alg(X',Fs,.1,4);
figure;plot(f,P,'.','MarkerSize',20)
box off
axis tight
set(gca,'FontSize',20)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
vline(5,'r')
ylabel('Power')
xlabel('Frequency')
xlim([.1 100])
ylim([0 1e-5])
yticks ''
f(1:15)


% state space traj of dynamics 
[coeff, score, latent] = pca(real(Xhat'));
figure;stem(cumsum(latent)./sum(latent))
figure;plot(score(:,1),score(:,2),'.')
xlabel('PC1')
ylabel('PC2')
title('State space traj of LFO-HG dynamics')
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off
set(gca,'FontSize',16)
xticks ''
yticks ''
figure;plot3(score(:,1),score(:,2),score(:,3),'.')
grid on

% looking to see if there are stable repeating dmd modes
bins = [.02:.02:5];
bin_counts=zeros(length(hg_mvmt),length(bins));
Xrecon=[];P_total=[];f_total=[];Phi_total=[];
for i=1:length(hg_mvmt)
    disp(i)
    temp = hg_mvmt{i};
    temp=temp(:,1:286);
    X=temp(:,sig_ch);
    [P, f ,Phi, lambda, Xhat, z0, Z]=dmd_alg(X',Fs,1.75,2.25);
    mean(Xhat(:))
    k=1;
    Xrecon(i,:,:) = Xhat(:,1:1500);
    Phi_total(i,:,:) = Phi;
    P_total(i,:) = P;
    f_total(i,:) = f;
    while f(k)<=5
        temp=f(k)>=bins;
        if sum(temp)>=1
            [aa bb]=find(temp==0);
            if abs(f(k) -  bins(bb(1))) < 0.05 % higher bin
                bin_counts(i,bb(1)) =1;
            else
                bin_counts(i,bb(1)-1) =1;
            end            
        end
        k=k+1;
    end
end

figure;imagesc(squeeze(real(mean(Xrecon,1))))
figure;plot(bins,sum(bin_counts,1),'.','MarkerSize',20);title('During Mvmt')
figure;plot(bins,smooth(sum(bin_counts,1)),'LineWidth',2)
set(gca,'FontSize',16)
set(gcf,'Color','w')
xlabel('Freq')
ylabel('Histogram across trials')
title('During Mvmt')

% play the movie
Xaa=squeeze(real(mean((Xrecon),1)))';
sig_ch1=sig_ch;
sig_ch1(16)=0;
M={};figure;k=1;
for i=1:2:size(Xaa,1)     
    disp(i)
    c_h = ctmr_gauss_plot(cortex,elecmatrix(sig_ch1,:),...
        zscore(Xaa(i,[1:3 5:end]))','lh');
     %view(-99,35)
    % caxis([-3 3])
    %e_h = el_add(elecmatrix(ch,:), 'color', 'b','msize',2);
    % To plot electrodes with numbers, use the following, as example:
    %e_h = el_add(elecmatrix(65:320,:), 'color', 'b', 'numbers', 1:286);
    % Or you can just plot them without labels with the default color
  %  e_h = el_add(elecmatrix(1:286,:),'numbers', 1:286); % only loading 48 electrode data
    set(gcf,'Color','w')    
    set(gca,'FontSize',20)
    title(num2str(i));
    M{k}=getframe;k=k+1;
    %pause(.01)
    clf
end


figure;
for i=1:1:size(M,2)
    imagesc(frame2im(M{i}));
    % text(7.2,7.2,num2str(i));
    pause(0.05)
    %title(num2str(i));    
end


figure;
vidObj = VideoWriter('EC189_HG_LFO_1Hz_Smoothing_New.avi');
vidObj.Quality=100;
open(vidObj);
for i=1:1:size(M,2)
    imagesc(frame2im(M{i}));
    % text(7.2,7.2,num2str(i));
    pause(0.0005)
    %title(num2str(i));
    writeVideo(vidObj,frame2im(M{i}));
end
close(vidObj);

% DMD test
temp = hg_rest{2};
temp=temp(:,1:286);
X=temp(:,sig_ch);
X1=zscore(X)';


% no. of stacks for Hankel
n = size(X1,1);
m = size(X1,2);
nstacks = 5 + ceil(2*m/n);

% construct the augmented, shift-stacked data matrices (Hankel matrix)
Xaug = [];
for st = 1:nstacks,
    Xaug = [Xaug; X1(:, st:end-nstacks+st)];
end
X1a = Xaug(:, 1:end-1);
X1dot = Xaug(:, 2:end);


[coeff,score,latent]=pca(X1');
temp = score(:,1:5)*coeff(:,1:5)';
X1=temp';
X1a = X1(:,1:end-1);
X1dot = X1(:,2:end);
A = X1dot*pinv(X1a);
[v d]=eigs(A);
%d=log(diag(d))/(1/Fs)/2/pi;

temp = hg_rest{2};
temp=temp(:,1:286);
X=temp(:,sig_ch);
[P, f ,Phi, lambda, Xhat, z0, Z,Phi_orig]=dmd_alg(X',Fs);

 


% plot the map of the sig. channels 

% combined
temp=sum(Wa1p,2);
temp(temp~=0)=1;
chI = find(temp~=0);
figure;
c_h = ctmr_gauss_plot(cortex,elecmatrix(chI,:),...
    temp(chI),'lh');
set(gcf,'Color','w')
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],...
    0,'lh');
e_h1 = el_add(elecmatrix(:,:), 'color', [1 1 1],'msize',2);
for j=1:length(chI)    
    e_h = el_add(elecmatrix(chI(j),:), 'color',[0.6882    0.6882    1.0000],'msize',6,'edgecol','k');
    e_h.LineWidth=1.25;
end
set(gcf,'Color','w')
set(gca,'FontSize',20)
view(-90,7)


% plotting HG amplitudes
figure;imagesc(zscore(X)')
colormap parula
caxis([-2 2])
set(gcf,'Color','w')
axis off

% finding the indices of the LFO component of HG amplitude 
index = (f>=0.1) .* (f<=4);
index = find(index==1);

% plotting the brain maps
for i=3:5
    ph = angle(Phi(:,i));
    phMap = linspace(-pi,pi,length(chI))';
    ChColorMap=parula(length(chI));
    val = abs(Phi(:,i));
    %scale val between 5 and 15
    figure;
    c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
    for j=1:length(val)
        ms = val(j)*(25);
        [aa bb]=min(abs(ph(j) - phMap));
        c=ChColorMap(bb,:);
        e_h = el_add(elecmatrix(chI(j),:), 'color', c,'msize',ms);
    end
    set(gcf,'Color','w');
end


% plotting the traces
for i=3:5
   figure;
   hold on
   plot(real(Z(i,:)),'b','LineWidth',2)
   plot(imag(Z(i,:)),'r','LineWidth',2)
   axis tight
   axis off
   set(gcf,'Color','w')
end



% plotting mode power
figure;stem(f(1:end),10*log10(P(1:end)),'k','LineWidth',2)
xlim([0 10])
set(gcf,'Color','w')
set(gca,'FontSize',20)
box off
set(gca,'LineWidth',2)

% plotting the results
total_pow=[];figure;hold on
for i=1:length(mvmt_pow)
    total_pow = [total_pow ;res_pow(i) mvmt_pow(i)];
    plot(i,res_pow(i),'.r','MarkerSize',15)
    plot(i+0.05,mvmt_pow(i),'.b','MarkerSize',15)
end
set(gcf,'Color','w')
xlabel('Trials')
ylabel('Power')
set(gca,'FontSize',16)
set(gca,'LineWidth',2)
%legend('Delta modes of HG during rest','Delta modes of HG during thumb flexion')

set(gcf,'PaperPositionMode','auto');
print('E:\DATA\ecog data\ECoG LeapMotion\Results\Figures for Paper Mat\EC189DMPAC',...
    '-dsvg','-r200');

