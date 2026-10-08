function parallel_check


if isempty(gcp('nocreate'))
    parpool('threads')
end


