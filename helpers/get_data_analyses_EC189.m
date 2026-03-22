%get_data_analyses_EC189



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
%lfp = lfp(:,bad_chI);

% median reference
m = median(lfp(:,bad_chI),2);
%lfp = lfp - median(lfp,2);
lfp=lfp-m;

