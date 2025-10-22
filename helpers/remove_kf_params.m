function files = remove_kf_params(files)

files1=[];
for i=1:length(files)
    if isempty(regexp(files{i},'kf_params'))
        files1=[files1;files(i)];
    end
end
files=files1;
