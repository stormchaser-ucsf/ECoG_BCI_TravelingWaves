% testing traveling waves in B3

clc;clear
%close all

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
cd(root_path)
load('ECOG_Grid_8596_000067_B3.mat')

% load B3 data from one trial
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20230511\HandImagined\113750\Imagined\Data0001.mat')

data_trial = (TrialData.BroadbandData');

% run power spectrum and 1/f stats on a single trial basis
task_state = TrialData.TaskState;
kinax = find(task_state==3);
data = cell2mat(data_trial(kinax));
spectral_peaks=[];
for i=1:size(data,2)
    x = data(:,i);    
    [Pxx,F] = pwelch(x,512,256,512,1e3);
    idx = logical((F>0) .* (F<=40));
    F1=F(idx);
    F1=log2(F1);
    power_spect = Pxx(idx);
    power_spect = log2(power_spect);
    [bhat p wh se ci t_stat]=robust_fit(F1,power_spect,1);
    x = [ones(length(F1),1) F1];
    yhat = x*bhat;

    %plot
    %figure;
    %plot(F1,power_spect);
    %hold on
    %plot(F1,yhat);    

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


% plotting it on the grid to see if there are any spatial clusters
bad_ch=[108 113 118];
osc_clus=[];
for f=2:40
    ff = [f-1 f+1];
    tmp=0;
    for j=1:length(spectral_peaks)
        if sum(j==bad_ch)==0
            freqs = spectral_peaks(j).freqs;
            for k=1:length(freqs)
                if ff(1) <= freqs(k)  && freqs(k) <= ff(2)
                    tmp=tmp+1;
                end                
            end
        end
    end
    osc_clus = [osc_clus tmp];
end

figure;plot(2:40,osc_clus)


