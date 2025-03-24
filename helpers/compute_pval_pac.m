function pval = compute_pval_pac(pac,alpha_phase,hg_alpha_phase)



pval=[];
rboot=[];
r = abs(mean(pac));
for iter=1:1000
    disp(iter)
    pac_boot=zeros(length(alpha_phase),253);
    parfor i = 1:length(alpha_phase)
        alp = alpha_phase{i};
        hg = hg_alpha_phase{i};

        % shuffle alpha phase
        for j=1:size(alp,2)
            idx = randperm(length(alp));
            alp(:,j) = alp(idx,j);
        end

        circ_diff = alp - hg;
        m = circ_mean(circ_diff);
        pac_boot(i,:)=m;
    end
    pac_boot = exp(1i*pac_boot);
    rboot(iter,:) = abs(mean(pac_boot));
end

pval = sum(rboot>r)/length(rboot);


end



