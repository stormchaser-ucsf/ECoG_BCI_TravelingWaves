function condn_data1=remove_bad_trials(condn_data)

condn_data1={};k=1;
for i=1:length(condn_data)
    if ~isempty(condn_data(i).neural)
        condn_data1(k).neural = condn_data(i).neural;
        condn_data1(k).targetID = condn_data(i).targetID;
        k=k+1;
    end
end
condn_data=condn_data1;

