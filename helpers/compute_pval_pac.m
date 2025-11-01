function [pval,rboot] = compute_pval_pac(pac,alpha_phase,hg_alpha_phase,varargin)


if isempty(varargin)
    b1=false;
else
    b1=true;
end


pval=[];
rboot=[];
r = abs(mean(pac));
parfor iter=1:6
    %fprintf('%d',iter)
    %disp(iter)
    if b1==false
        pac_boot=zeros(length(alpha_phase),253);
    else
        pac_boot=zeros(length(alpha_phase),128);
    end

    % new... vectorization
    % function for matrix difference        
    for i = 1:numel(alpha_phase)                
        tmp = shuffle_columns(alpha_phase{i}) - hg_alpha_phase{i};  
        m = angle(mean(exp(1i*tmp)));
        pac_boot(i,:) = m;
    end

    

    % for i = 1:length(alpha_phase)
    %     alp = alpha_phase{i};
    %     hg = hg_alpha_phase{i};        
    % 
    %     % shuffle alpha phase        
    %     for j=1:size(alp,2)
    %         idx = randperm(size(alp,1));
    %         alp(:,j) = alp(idx,j);
    %     end
    % 
    % 
    %     circ_diff = alp - hg;
    %     m = circ_mean(circ_diff);
    %     pac_boot(i,:)=m;
    % end
    pac_boot = exp(1i*pac_boot);
    rboot(iter,:) = abs(mean(pac_boot));
end

pval = sum(rboot>r)/size(rboot,1);


end






