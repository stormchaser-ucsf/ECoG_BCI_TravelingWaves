function [mahab_dist] = get_mahab_dist(files,varargin)


if length(varargin)==0
    b1=false;
else
    b1=true;
end

if b1==false
    bad_ch=[108 113 118];
    good_ch=ones(256,1);
    good_ch(bad_ch)=0;
else
    good_ch=(ones(128,1));
end

D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
D8=[];
D9=[];
D10=[];
D11=[];
D12=[];

for ii=1:length(files)
    disp(ii/length(files)*100)
    loaded=1;
    try
        load(files{ii})
    catch
        loaded=0;
    end

    if loaded==1

        kinax1 = find(TrialData.TaskState==1);
        kinax2 = find(TrialData.TaskState==2);
        kinax3 = find(TrialData.TaskState==3);
        kinax4 = find(TrialData.TaskState==4);
        tid = TrialData.TargetID;

        data1 = cell2mat(TrialData.SmoothedNeuralFeatures(kinax1))';
        l1 =  size(data1,1);
        data2 = cell2mat(TrialData.SmoothedNeuralFeatures(kinax2))';
        l2 =  size(data2,1);
        data4 = cell2mat(TrialData.SmoothedNeuralFeatures(kinax4))';
        l4 = size(data4,1);
        data3 = cell2mat(TrialData.SmoothedNeuralFeatures(kinax3))';
        l3 = size(data3,1);

        % get the hG signals alone
        hg =  data3(:,1537:end);

        % remove bad channels
        hg = hg(:,logical(good_ch));

        % store
        if tid==1
            D1 = [D1 ; hg];

        elseif tid==2
            D2 = [D2 ; hg];

        elseif tid==3
            D3 = [D3 ; hg];

        elseif tid==4
            D4 = [D4 ; hg];

        elseif tid==5
            D5 = [D5 ; hg];

        elseif tid==6
            D6 = [D6 ; hg];

        elseif tid==7
            D7 = [D7 ; hg];

        elseif tid==8
            D8 = [D8 ; hg];

        elseif tid==9
            D9 = [D9 ; hg];

        elseif tid==10
            D10 = [D10 ; hg];

        elseif tid==11
            D11 = [D11 ; hg];

        elseif tid==12
            D12 = [D12 ; hg];
        end
    

    end
end

condn_data{1} = D1;
condn_data{2} = D2;
condn_data{3} = D3;
condn_data{4} = D4;
condn_data{5} = D5;
condn_data{6} = D6;
condn_data{7} = D7;
condn_data{8} = D8;
condn_data{9} = D9;
condn_data{10} = D10;
condn_data{11} = D11;
condn_data{12} = D12;


%[D] = mahal2_full(condn_data);


% now doing it channel by channel 
D_chan=[];
for ch=1:253 %channels
    D=zeros(12);
    for j=1:length(condn_data)
        tmp = condn_data{j};
        tmp = tmp(:,ch);
        for k=j+1:length(condn_data)
            tmp1 = condn_data{k};
            tmp1 = tmp1(:,ch);
            c = (var(tmp) + var(tmp1))/2;
            m = abs(mean(tmp) - mean(tmp1));
            d = m/sqrt(c);
            D(j,k) = d;
            D(k,j) = d;
        end
    end
    D = squareform((D'));
    D_chan(ch) = mean(D);
end

mahab_dist = D_chan;









