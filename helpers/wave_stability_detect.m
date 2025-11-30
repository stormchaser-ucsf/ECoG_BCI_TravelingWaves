function [wav_det,wav_st,wav_stp] = wave_stability_detect(tmp,thresh)

if nargin<2
    thresh=0;
end

wav_det=[];
wav_st=[];wav_stp=[];
done=0;
k=1;
while ~done
    start_idx=k;
    while tmp(k)>thresh && k<length(tmp)
        % march forward in time
        k=k+1;                  
    end
    end_idx=k;  
    
    if end_idx == start_idx
        k=k+1;
    end
    
    len = end_idx-1-start_idx;
    if len >= 3 %50Hz -> greater than 60ms
        wav_det = [wav_det len];
        wav_st = [wav_st start_idx];
        wav_stp = [wav_stp end_idx-1];
    end

    if k==length(tmp)
        done=1;
    end
end

% 
% 
% figure;plot(tmp)
% hline(0)
% vline(wav_st,'g')
% vline(wav_stp,'r')
