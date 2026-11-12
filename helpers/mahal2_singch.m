function [mahab_dist] = mahal2_singch(a,b)


ch_len = size(a,2);
mahab_dist = [];
parfor ch=1:ch_len %channels
    D=zeros(12);
    tmp = a(:,ch);
    tmp1 = b(:,ch);
    c = (var(tmp) + var(tmp1))/2;
    m = abs(mean(tmp) - mean(tmp1));
    d = m/sqrt(c);
    mahab_dist(ch) = d;
end


