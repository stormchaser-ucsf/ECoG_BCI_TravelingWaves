function [pval1,pval2] = test_hG_wave_nonwave_dec_acc(boot,subj)

% compare to actual data
clear acc_nonwave acc_wave
if strcmp(subj,'B1')
    load('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/hg_wave_nonwave_MLP_3DArrow_CL_v2.mat')
elseif strcmp(subj,'B3')
    load('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/hg_wave_nonwave_MLP_3DArrow_CL_v2.mat')
elseif strcmp(subj,'B6')
    %load hg_wave_nonwave_MLP_3DArrow_CL_AllData_v4_AllFolders
    %load hg_wave_nonwave_MLP_3DArrow_CL_AllData_v3_AllFolders
    load('/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/hg_wave_nonwave_MLP_3DArrow_CL_AllData_v5_AllFolders.mat')    
end

res=[];
for i=1:size(acc_nonwave,1)
    tmp=squeeze((acc_nonwave(i,:,:)));
    tmp1=squeeze((acc_wave(i,:,:)));
    res(i,:) = [ mean(diag(tmp)) mean(diag(tmp1))];
end
stat = mean(res(:,1)-res(:,2));

boot_val=[];
for i=1:10000
    idx = randi(length(boot),1,size(acc_nonwave,1));
    boot_val(i) = mean(boot(idx));
end
% figure;
% hist(boot_val)
% vline(stat)
pval1=1-sum(stat>=abs(boot_val))/length(boot_val);
pval2=1-sum(stat>=(boot_val))/length(boot_val);

end