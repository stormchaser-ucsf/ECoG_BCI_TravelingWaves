% testing traveling waves in B3

clc;clear
%close all

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
cd(root_path)
load('ECOG_Grid_8596_000067_B3.mat')

% add the circ stats toolbox
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\wave-matlab-master\wave-matlab-master'))
addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves')
imaging_B3;close all

%% OPEN LOOP OSCILLATION CLUSTERS
% get all the files from a particular day
%filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20230511\HandImagined';
%filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20230518\HandOnline';
filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20230223\Robot3DArrow';

files = findfiles('.mat',filepath,1)';

files1=[];
for i=1:length(files)
    if isempty(regexp(files{i},'kf_params'))
        files1=[files1;files(i)];
    end
end
files=files1;
files=files(1:100);

% if you want to examine hG waves
Fs=1000;
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',Fs);

bad_ch=[108 113 118];
osc_clus=[];
stats=[];
for ii=1:length(files)
    disp(ii/length(files)*100)
    load(files{ii})
    data_trial = (TrialData.BroadbandData');

    % run power spectrum and 1/f stats on a single trial basis
    task_state = TrialData.TaskState;
    l2 = find(task_state==2);
    kinax = find(task_state==3);
    data = cell2mat(data_trial(kinax));
    
    % extract hG alone
    %data=abs(hilbert(filtfilt(bpFilt,data)));

    spectral_peaks=[];
    stats_tmp=[];
    parfor i=1:size(data,2)
        x = data(:,i);
        [Pxx,F] = pwelch(x,1024,512,1024,1e3);
        idx = logical((F>0) .* (F<=40));
        F1=F(idx);
        F1=log2(F1);
        power_spect = Pxx(idx);
        power_spect = log2(power_spect);
        %[bhat p wh se ci t_stat]=robust_fit(F1,power_spect,1);
        tb=fitlm(F1,power_spect,'RobustOpts','on');
        stats_tmp = [stats_tmp tb.Coefficients.pValue(2)];
        bhat = tb.Coefficients.Estimate;
        x = [ones(length(F1),1) F1];
        yhat = x*bhat;

        %plot
%         figure;
%         plot(F1,power_spect,'LineWidth',1);
%         hold on
%         plot(F1,yhat,'LineWidth',1);

        % get peaks in power spectrum at specific frequencies
        power_spect = zscore(power_spect - yhat);
        [aa bb]=findpeaks(power_spect);
        peak_loc = bb(find(power_spect(bb)>1));
        freqs = 2.^F1(peak_loc);
        pow = power_spect(peak_loc);

        %store
        spectral_peaks(i).freqs = freqs;
        spectral_peaks(i).pow = pow;
    end


    % getting oscillation clusters
    osc_clus_tmp=[];
    for f=2:40
        ff = [f-1 f+1];
        tmp=0;ch_tmp=[];
        for j=1:length(spectral_peaks)
            if sum(j==bad_ch)==0
                freqs = spectral_peaks(j).freqs;
                for k=1:length(freqs)
                    if ff(1) <= freqs(k)  && freqs(k) <= ff(2)
                        tmp=tmp+1;
                        ch_tmp = [ch_tmp j];
                    end
                end
            end
        end
        osc_clus_tmp = [osc_clus_tmp tmp];
    end
    osc_clus = [osc_clus;osc_clus_tmp];
end


% get spectral peaks
pks=[];
for i=1:length(spectral_peaks)
    pks=[pks ;spectral_peaks(i).freqs];
end
figure;hist(pks,16)

% plot oscillation clusters
f=2:40;
figure;
hold on
plot(f,osc_clus,'Color',[.5 .5 .5 .5],'LineWidth',.5)
plot(f,median(osc_clus,1),'b','LineWidth',2)

% get all the electrodes with peak between 8.0Hz and 10Hz
ch_idx=[];
for i=1:length(spectral_peaks)
    if sum(i==bad_ch)==0
        f = spectral_peaks(i).freqs;
        if sum( (f>=8) .* (f<=10) ) >= 1
            ch_idx=[ch_idx i];
        end
    end
end
length(ch_idx)/253
I = zeros(256,1);
I(ch_idx)=1;
figure;imagesc(I(ecog_grid))

% get all electrodes within 16 and 20Hz
ch_idx=[];
for i=1:length(spectral_peaks)
    if sum(i==bad_ch)==0
        f = spectral_peaks(i).freqs;
        if sum( (f>=20) .* (f<=26) ) >= 1
            ch_idx=[ch_idx i];
        end
    end
end
length(ch_idx)/253
I = zeros(256,1);
I(ch_idx)=1;
figure;imagesc(I(ecog_grid))


%% ON TRIAL AVERAGED POWER SPECTRUM
% 
% 
% clc;clear
% %close all
% 
% root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
% cd(root_path)
% load('ECOG_Grid_8596_000067_B3.mat')
% 
% % get all the files from a particular day
% %filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20230511\HandImagined';
% filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20230511\HandOnline';
% 
% files = findfiles('.mat',filepath,1)';
% 
% files1=[];
% for i=1:length(files)
%     if isempty(regexp(files{i},'kf_params'))
%         files1=[files1;files(i)];
%     end
% end
% files=files1;
% 
% bad_ch=[108 113 118];
% osc_clus=[];
% pow_spec = [];
% for ii=1:length(files)
%     disp(ii/length(files)*100)
%     load(files{ii})
%     data_trial = (TrialData.BroadbandData');
% 
%     % run power spectrum and 1/f stats on a single trial basis
%     task_state = TrialData.TaskState;
%     kinax = find(task_state==3);
%     data = cell2mat(data_trial(kinax));
%     spectral_peaks=[];
%     tmp=[];ff=[];
%     parfor i=1:size(data,2)
%         x = data(:,i);
%         [Pxx,F] = pwelch(x,1024,512,1024,1e3);
%         tmp = [tmp Pxx];
%         ff = [ ff F];
%     end
%     pow_spec(ii,:,:) = tmp;
% end
% 
% spectral_peaks=[];
% F = ff(:,end);
% for i=1:size(pow_spec,3)
%     Pxx =  squeeze(mean(pow_spec(:,:,i),1));
%     idx = logical((F>0) .* (F<=40));
%     F1=F(idx);
%     F1=log2(F1);
%     power_spect = Pxx(idx)';
%     power_spect = log2(power_spect);
%     %[bhat p wh se ci t_stat]=robust_fit(F1,power_spect,1);
%     tb=fitlm(F1,power_spect,'RobustOpts','on');
%     bhat = tb.Coefficients.Estimate;
%     x = [ones(length(F1),1) F1];
%     yhat = x*bhat;
% 
%     %plot
%     %     figure;
%     %     plot(F1,power_spect);
%     %     hold on
%     %     plot(F1,yhat);
% 
%     % get peaks in power spectrum at specific frequencies
%     power_spect = zscore(power_spect - yhat);
%     [aa bb]=findpeaks(power_spect);
%     peak_loc = bb(find(power_spect(bb)>1.0));
%     freqs = 2.^F1(peak_loc);
%     pow = power_spect(peak_loc);
% 
%     %store
%     spectral_peaks(i).freqs = freqs;
%     spectral_peaks(i).pow = pow;
% end
% 
% % getting oscillation clusters
% osc_clus=[];
% for f=2:40
%     fff = [f-1 f+1];
%     tmp=0;ch_tmp=[];
%     for j=1:length(spectral_peaks)
%         if sum(j==bad_ch)==0
%             freqs = spectral_peaks(j).freqs;
%             for k=1:length(freqs)
%                 if fff(1) <= freqs(k)  && freqs(k) <= fff(2)
%                     tmp=tmp+1;
%                     ch_tmp = [ch_tmp j];
%                 end
%             end
%         end
%     end
%     osc_clus = [osc_clus tmp];
% end
% 
% 
% % plot oscillation clusters
% f=2:40;
% figure;
% hold on
% plot(f,osc_clus,'LineWidth',2)

%% look at a single trial example of traveling waves

ii=13;
load(files{ii})
data_trial = (TrialData.BroadbandData');
task_state = TrialData.TaskState;
[aa bb]=unique(task_state);
states=[];
len_states=[];
for i=1:length(bb)-1
    states(:,i) = [bb(i) bb(i+1)-1]';
    tmp = cell2mat(data_trial(bb(i):(bb(i+1)-1)));
    len_states = [len_states size(tmp,1)];
end

% filter in the 8 to 10Hz range
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',100);
%fvtool(bpFilt)

data = cell2mat(data_trial);

% down sample to 50Hz
data = resample(data,100,1e3);


data = filter(bpFilt,data);
data_ang = angle(hilbert(data));



% extract wave at specific time points 
len_states1 = cumsum(len_states);
len_states1 = round(len_states1/10)
%data = data(len_states1(2):len_states1(end),:);
%data_ang = data_ang(len_states1(2):len_states1(end),:);

% have to now cut the ecog_grid down to size

% plot it as a movie
figure;
v = VideoWriter('CL_TravWave_B3_Hand_Segment.avi');
v.FrameRate=6;
open(v);
for i=1:1:size(data,1)
    tmp = data_ang(i,:);
    tmp = cos(tmp);
    imagesc(tmp(ecog_grid));
    colormap hot
    caxis([-1 1])
    colorbar
    title('Cosine of phase in 8-10Hz')
    axis off

    textboxHandle = uicontrol('Style', 'text', 'Position', [0, 0, 200, 40]);
    UIControl_FontSize_bak = get(0, 'DefaultUIControlFontSize');
    set(0, 'DefaultUIControlFontSize', 12);

    if i>len_states1(2)
        txt = ['Active Control:  ' num2str(i*10) 'ms'];
    elseif i<len_states1(1)
        txt = ['State 1:  ' num2str(i*10) 'ms'];
    elseif i>=len_states1(1) && i <= len_states1(2)
        txt = ['State 2:  ' num2str(i*10) 'ms'];
    elseif i>=len_states1(end)
        txt = ['State 4:  ' num2str(i*10) 'ms'];
    end

    newText = sprintf(txt);
    set(textboxHandle, 'String', newText);
    cc=getframe(gcf);
    writeVideo(v, cc);
    %pause(0.05)
end
close(v)

% have to arrange the data in the format as above i.e, channel 1 is top
% right corner and increases going left.
channel_layout = [];
for i=1:23:253
    channel_layout = [channel_layout;i:i+22];
end
channel_layout = fliplr(channel_layout);


[c,s,l]=pca(elecmatrix);
locs = s(:,1:2);
X = locs(:,1);
Y = locs(:,2);
pred = locs;
pred(:,1) = pred(:,1)./max(pred(:,1));
pred(:,2) = pred(:,2)./max(pred(:,2));

% 
% % using old school method
% pred = [];
% for i=1:11
%     pred = [pred; [ (1:23)' repmat(i,23,1)]];
% end
% pred(:,1) = pred(:,1)./max(pred(:,1));
% pred(:,2) = pred(:,2)./max(pred(:,2));


% doing circular linear analyses at each time point
corr_val=[];
alp_range = 0:0.5:360;
r_range = (0.01:0.25:25)';
angles = [];
for i=1:size(data,1)


    disp(i*100/size(data,1))
    
    tmp = data_ang(i,:);
    tmp = tmp(ecog_grid);

    theta_orig = tmp;

    tmp = flipud(tmp');
    tmp = tmp(:);

    

    theta = tmp;
    theta = wrapTo2Pi(theta);
    rval=[];

    % vectorizing   

    as = r_range * cosd(alp_range);
    len = size(as);
    as=as(:);
    bs = r_range * sind(alp_range);
    bs = bs(:);

    theta_hat = pred * ([as';bs']);
    y = repmat(theta,1,size(theta_hat,2)) - theta_hat;
    r1 = mean(cos(y));
    r2 = mean(sin(y));
    rtmp = (r1.^2 + r2.^2).^(0.5);
    rval = reshape(rtmp,[len]);

    % get the best regression parameters
    [aa bb]=find(rval==max(rval(:)));
    if length(aa)>1
        aa=aa(1);
        bb=bb(1);
    end
    alp_hat=alp_range;
    r_hat=  r_range;
    alp_hat = alp_hat(bb);
    r_hat = r_hat(aa);
    a = r_hat*cosd(alp_hat);
    b = r_hat*sind(alp_hat);

    % get the phase offset
    theta_hat = wrapTo2Pi(pred*([a;b]));
    y1 = sum(sin(theta-theta_hat));
    y2 = sum(cos(theta-theta_hat));
    phi = atan2(y1,y2);

    % final reconstruction
    theta_hat = wrapTo2Pi(theta_hat + phi);

     % get circular correlation
    [rho pval] = circ_corrcc(theta_hat(:), theta(:));
    %disp(rho)

    % store results
    corr_val(i)=rho;
    angles(i) = alp_hat;
    

%     % rearranging as a 2D array
%     theta_hat = reshape(theta_hat,[size(theta_orig)]);
%     figure;
%     subplot(1,2,1)
%     imagesc(theta_orig)
%     axis tight
%     title('Simulated Original')
%     subplot(1,2,2)
%     imagesc(theta_hat)
%     title('Recon from circular linear regression')
%     axis tight 



end

tt=linspace(0,size(data,1)*10,size(data,1));
figure;
subplot(2,1,1)
plot(tt,corr_val.^2,'LineWidth',1)
plot_beautify
xlim([0 size(data,1)*10])
ylim([0 0.75])
xlabel('Time (ms)')
vline(len_states1*10,'--r')
ylabel('Wave Strength')
sgtitle('OL trial')

subplot(2,1,2);
plot(tt,angles,'LineWidth',1)
plot_beautify
xlim([0 size(data,1)*10])
xlabel('Time (ms)')
vline(len_states1*10,'--r')
ylabel('Wave Angle')


% plotting stability of traveling wave epochs 
stab=[];
for i=2:length(corr_val)
    tmp = cosd(angles(i)) + 1i*sind(angles(i));
    tmpm1 = cosd(angles(i-1)) + 1i*sind(angles(i-1));    
    stab(i) = abs((corr_val(i)*tmp) - (corr_val(i-1)*tmpm1));
end
figure;plot(zscore(stab))

% plotting stability of phase angles 

%% LOOKING AT EXPANDING AND CONTRACTING WAVES

% continuing work on the trial loaded from above
ii=1;
load(files{ii})
data_trial = (TrialData.BroadbandData');
task_state = TrialData.TaskState;
[aa bb]=unique(task_state);
states=[];
len_states=[];
for i=1:length(bb)-1
    states(:,i) = [bb(i) bb(i+1)-1]';
    tmp = cell2mat(data_trial(bb(i):(bb(i+1)-1)));
    len_states = [len_states size(tmp,1)];
end

% filter in the 8 to 10Hz range
Fs=100;
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',Fs);
%fvtool(bpFilt)

data = cell2mat(data_trial);

% down sample to 50Hz
data = resample(data,100,1e3);

% filter
data = filter(bpFilt,data);

% datacube
dc=[];
for i=1:size(data,1)
    t = data(i,:)';
    t = t(ecog_grid);
    dc(:,:,i) = t;
end

xf = zscore_independent( dc );

% form analytic signal
xph = analytic_signal( xf );

% calculate instantaneous frequency 
[wt,signIF] = instantaneous_frequency( xph, Fs );


% calculate phase gradient
pixel_spacing = 1; %a.u.
[pm,pd,dx,dy] = phase_gradient_complex_multiplication( xph, pixel_spacing, signIF );


% plot resulting vector field
plot_vector_field( exp( 1i .* pd(:,:,200) ), 1 );

[XX,YY] = meshgrid( 1:size(pd,2), 1:size(pd,1) );

% expanding wave metrics
res=[];
for i=1:size(pd,3)
    %disp(i)
    %plot_vector_field( exp( 1i .* pd(:,:,i) ), 1 );
    tmp=exp( 1i .* pd(:,:,i) );
    M = real( exp( 1i * angle(tmp) ) ); N = imag( exp( 1i * angle(tmp) ) );
    M = smoothn(M);
    N = smoothn(N);
    %[cl,c]= curl(XX,YY,M,N);
    [cl]= divergence(XX,YY,M,N);
    pl = squeeze(angle(xph(:,:,i)));
    [cc,pv] = phase_correlation_distance( pl, cl,[]);
    res(i)=cc;
    %title(['Correlation of ' num2str(cc)]);
end

tt=linspace(0,size(data,1)*10,size(data,1));
figure;plot(tt,smooth(res))
vline(cumsum(len_states),'--r')



%% LOOKING AT ROTATING WAVES

% continuing work on the trial loaded from above
ii=6;
load(files{ii})
data_trial = (TrialData.BroadbandData');
task_state = TrialData.TaskState;
[aa bb]=unique(task_state);
states=[];
len_states=[];
for i=1:length(bb)-1
    states(:,i) = [bb(i) bb(i+1)-1]';
    tmp = cell2mat(data_trial(bb(i):(bb(i+1)-1)));
    len_states = [len_states size(tmp,1)];
end

% filter in the 8 to 10Hz range
Fs=100;
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',Fs);
%fvtool(bpFilt)

data = cell2mat(data_trial);

% down sample to 50Hz
data = resample(data,100,1e3);

% filter
data = filter(bpFilt,data);

% datacube
dc=[];
for i=1:size(data,1)
    t = data(i,:)';
    t = t(ecog_grid);
    dc(:,:,i) = t;
end

xf = zscore_independent( dc );

% form analytic signal
xph = analytic_signal( xf );

% calculate instantaneous frequency 
[wt,signIF] = instantaneous_frequency( xph, Fs );


% calculate phase gradient
pixel_spacing = 1; %a.u.
[pm,pd,dx,dy] = phase_gradient_complex_multiplication( xph, pixel_spacing, signIF );


% plot resulting vector field
plot_vector_field( exp( 1i .* pd(:,:,97) ), 1 );

[XX,YY] = meshgrid( 1:size(pd,2), 1:size(pd,1) );

% expanding wave metrics
res=[];
for i=1:size(pd,3)
    %disp(i)
    %plot_vector_field( exp( 1i .* pd(:,:,i) ), 1 );
    tmp=exp( 1i .* pd(:,:,i) );
    M = real( exp( 1i * angle(tmp) ) ); N = imag( exp( 1i * angle(tmp) ) );
    M = smoothn(M);
    N = smoothn(N);
    [cl,c]= curl(XX,YY,M,N);
    pl = squeeze(angle(xph(:,:,i)));
    %[cc,pv,center_point] = phase_correlation_rotation( pl, cl,[],signIF);
    [cc,pv] = phase_correlation_distance( pl, source, spacing );
    res(i)=cc;
    %title(['Correlation of ' num2str(cc)]);
end

tt=linspace(0,size(data,1)*10,size(data,1));
figure;plot(tt,smooth(res))
vline(cumsum(len_states),'--r')






