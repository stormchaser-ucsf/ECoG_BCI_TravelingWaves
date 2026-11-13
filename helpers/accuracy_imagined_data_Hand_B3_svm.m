function [acc,train_permutations,acc_bin,bino_pdf,bino_pdf_chance] = ...
    accuracy_imagined_data_Hand_B3_svm(condn_data, iterations)

num_trials = length(condn_data);
train_permutations = zeros(num_trials,iterations)';
acc_permutations=[];
bino_pdf={};
bino_pdf_chance={};
iterations = length(condn_data);
for iter = 1:iterations
    disp(['Processing trial ' num2str(iter)])

    % chk=0;chk1=0;xx=0;
    % while chk==0 || chk1==0
    %     % split of 70% training, 15% validation and 15% testing
    %     train_idx = randperm(num_trials,round(0.70*num_trials));
    %     test_idx = ones(num_trials,1);
    %     test_idx(train_idx) = 0;
    %     test_idx = find(test_idx==1);
    % 
    %     chk= 12 == length(unique([condn_data(train_idx).targetID]));
    %     chk1= 12 == length(unique([condn_data(test_idx).targetID]));
    %     xx=xx+1;
    % end
    % 
    % train_permutations(iter,train_idx) = train_permutations(iter,train_idx)+1;

    % split into train and test set leave one trial out
    test_idx = iter;
    train_idx =  ones(iterations,1);
    train_idx(iter)=0;
    train_idx = find(train_idx==1);

    % build a MLP from the training data
    train_data = condn_data(train_idx);
    test_data = condn_data(test_idx);

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
    for i=1:length(train_data)
        temp = train_data(i).neural;
        if train_data(i).targetID == 1
            D1 = [D1 temp];
        elseif train_data(i).targetID == 2
            D2 = [D2 temp];
        elseif train_data(i).targetID == 3
            D3 = [D3 temp];
        elseif train_data(i).targetID == 4
            D4 = [D4 temp];
        elseif train_data(i).targetID == 5
            D5 = [D5 temp];
        elseif train_data(i).targetID == 6
            D6 = [D6 temp];
        elseif train_data(i).targetID == 7
            D7 = [D7 temp];
        elseif train_data(i).targetID == 8
            D8 = [D8 temp];
        elseif train_data(i).targetID == 9
            D9 = [D9 temp];
        elseif train_data(i).targetID == 10
            D10 = [D10 temp];
        elseif train_data(i).targetID == 11
            D11 = [D11 temp];
        elseif train_data(i).targetID == 12
            D12 = [D12 temp];
        end
    end


    clear condn_data1
    idx=1;
    condn_data1{1}=[D1(idx:end,:) ]';
    condn_data1{2}= [D2(idx:end,:)]';
    condn_data1{3}=[D3(idx:end,:)]';
    condn_data1{4}=[D4(idx:end,:)]';
    condn_data1{5}=[D5(idx:end,:)]';
    condn_data1{6}=[D6(idx:end,:)]';
    condn_data1{7}=[D7(idx:end,:)]';
    condn_data1{8}=[D8(idx:end,:)]';
    condn_data1{9}=[D9(idx:end,:)]';
    condn_data1{10}=[D10(idx:end,:)]';
    condn_data1{11}=[D11(idx:end,:)]';
    condn_data1{12}=[D12(idx:end,:)]';

    %     % data augmentation -> extract random 5 snippets 400 times
    %     for i=1:length(condn_data1)
    %         tmp = condn_data1{i};
    %         len = size(tmp,1)*4;
    %
    %         % delta
    %         tmp_delta = tmp(:,1:253);
    %         %         new_tmp=[];
    %         %         for j=1:len
    %         %             idx= randperm(size(tmp_delta,1),5);
    %         %             xx = mean(tmp_delta(idx,:),1);
    %         %             new_tmp =[new_tmp;xx];
    %         %         end
    %         %         tmp_delta = [tmp_delta;new_tmp];
    %         C = cov(tmp_delta);
    %         if rank(C)<size(tmp_delta,2)
    %             C = C + 1e-9*eye(size(C));
    %         end
    %         C12 = chol(C);
    %         m = mean(tmp_delta,1);
    %         x = randn(len,size(C,1));
    %         new_delta = x*C12+m;
    %
    %         % beta
    %         tmp_beta = tmp(:,254:2*253);
    %         %         new_tmp=[];
    %         %         for j=1:len
    %         %             idx= randperm(size(tmp_beta,1),5);
    %         %             xx = mean(tmp_beta(idx,:),1);
    %         %             new_tmp =[new_tmp;xx];
    %         %         end
    %         %         tmp_beta = [tmp_beta;new_tmp];
    %         C = cov(tmp_beta);
    %         if rank(C)<size(tmp_beta,2)
    %             C = C + 1e-9*eye(size(C));
    %         end
    %         C12 = chol(C);
    %         m = mean(tmp_beta,1);
    %         x = randn(len,size(C,1));
    %         new_beta = x*C12+m;
    %
    %         % hg
    %         tmp_hg = tmp(:,507:end);
    %         %         new_tmp=[];
    %         %         for j=1:len
    %         %             idx= randperm(size(tmp_hg,1),5);
    %         %             xx = mean(tmp_hg(idx,:),1);
    %         %             new_tmp =[new_tmp;xx];
    %         %         end
    %         %         tmp_hg = [tmp_hg;new_tmp];
    %         C = cov(tmp_hg);
    %         if rank(C)<size(tmp_hg,2)
    %             C = C + 1e-9*eye(size(C));
    %         end
    %         C12 = chol(C);
    %         m = mean(tmp_hg,1);
    %         x = randn(len,size(C,1));
    %         new_hg = x*C12+m;
    %
    %         %new_data = [tmp_delta tmp_beta tmp_hg];
    %         new_data = [new_delta new_beta new_hg];
    %         tmp=[tmp;new_data];
    %         condn_data1{i}=tmp;
    %     end


    % organize data
    N=[];
    T1=[];
    for i=1:length(condn_data1)
        tmp=condn_data1{i};
        N = [N tmp'];
        T1 = [T1;i*ones(size(tmp,1),1)];
    end
    T = zeros(size(T1,1),length(condn_data1));
    for i=1:length(condn_data1)
        [aa bb]=find(T1==i);
        T(aa(1):aa(end),i)=1;
    end

    % train MLP
    % net = patternnet([120  ]) ;
    % net.performParam.regularization=0.2;
    % net.divideParam.trainRatio = 0.80;
    % net.divideParam.valRatio = 0.2;
    % net.divideParam.testRatio = 0;
    % net.trainParam.showWindow = 1;
    % net.trainParam.epochs = 500;
    % net = train(net,N,T','useParallel','no','UseGPU','yes');

    % train SVM
    % N: [nFeatures x nSamples]
    % T: [nClasses x nSamples] (one-hot encoded)

    T = T';
    %X = N';
    X = gpuArray(single(N'));
    [~, Y] = max(T, [], 1);
    Y = Y';

    % Standardize each feature across samples
    % mu = mean(X, 1);
    % sigma = std(X, 0, 1);
    % sigma(sigma == 0) = 1;
    % X = (X - mu) ./ sigma;

    % Train multiclass linear SVM
    %t = templateSVM('KernelFunction', 'linear', 'Standardize', true);
    t = templateLinear('Learner', 'svm', ...
        'Solver', 'lbfgs', ...
        'Regularization', 'ridge', ...
        'IterationLimit', 1000, ...
        'BetaTolerance', 1e-5, ...
        'GradientTolerance', 1e-6);

    mdl = fitcecoc(X, Y, 'Learners', t, 'Coding', 'onevsall');

    % test data on held out trial 
    acc = nan(length(condn_data1));
    acc_bin = nan(length(condn_data1));
    features = test_data.neural;
    Ntest = features;
    Xtest = gpuArray(single(Ntest))';
    %Xtest = (Xtest - mu) ./ sigma;
    [Ypred, scores] = predict(mdl, Xtest);
    Ypred = gather(Ypred);
    scores = gather(scores);
    out = scores';


    %out(out<0.4)=0; % thresholding
    [prob,idx] = max(out); % getting the decodes
    decodes=idx;
    %decodes = mode_filter(idx,length(condn_data1)); % running it through a 5 sample mode filter
    decodes_sum=[];
    for ii=1:length(condn_data1)
        decodes_sum(ii) = sum(decodes==ii);
    end
    [aa bb]=max(decodes_sum);

    if sum(aa==decodes_sum)==1
        if test_data.targetID == bb
            acc_permutations(iter) = 1;
        else
            acc_permutations(iter) = 0;
        end
    else
        disp(['error trial ' num2str(i)])
        xx=mean(out,2);
        [aa bb]=max(xx);
        if test_data.targetID == bb
            acc_permutations(iter) = 1;
        else
            acc_permutations(iter) = 0;
        end
    end

   
   
   % acc_bin_permutations(iter,:,:) = acc_bin1;
    



    % % test it out on the held out trials using a mode filter
    % acc = zeros(length(condn_data1));
    % acc_bin = zeros(length(condn_data1));
    % for i=1:length(test_data)
    %     features = test_data(i).neural;
    %     if ~isempty(features) && test_data(i).targetID<=length(condn_data1)
    %         Ntest = features;
    %         Xtest = gpuArray(single(Ntest))';
    %         %Xtest = (Xtest - mu) ./ sigma;
    %         [Ypred, scores] = predict(mdl, Xtest);
    %         Ypred = gather(Ypred);
    %         scores = gather(scores);
    %         out = scores';
    % 
    % 
    %         %out(out<0.4)=0; % thresholding
    %         [prob,idx] = max(out); % getting the decodes
    %         decodes=idx;
    %         %decodes = mode_filter(idx,length(condn_data1)); % running it through a 5 sample mode filter
    %         decodes_sum=[];
    %         for ii=1:length(condn_data1)
    %             decodes_sum(ii) = sum(decodes==ii);
    %         end
    %         [aa bb]=max(decodes_sum);
    % 
    %         if sum(aa==decodes_sum)==1
    %             acc(test_data(i).targetID,bb) = acc(test_data(i).targetID,bb)+1; % trial level
    %         else
    %             disp(['error trial ' num2str(i)])
    %             xx=mean(out,2);
    %             [aa bb]=max(xx);
    %             acc(test_data(i).targetID,bb) = acc(test_data(i).targetID,bb)+1; % trial level
    %         end
    % 
    %         % bin level
    %         for j=1:length(idx)
    %             acc_bin(test_data(i).targetID,idx(j)) = ...
    %                 acc_bin(test_data(i).targetID,idx(j))+1;
    %         end
    % 
    %         %acc(test_data(i).targetID,bb) = acc(test_data(i).targetID,bb)+1;
    %     end
    % end
    %     for i=1:size(acc,1)
    %         acc1(i,:) = acc(i,:)/sum(acc(i,:));
    %     end
    %     %acc1
    %     acc_permutations(iter,:,:) = acc1;
    % 
    % % binomial distribution around the actual results, that acc. chance or
    % % worse than chance
    % n = length(test_data);
    % succ = sum(diag(acc));
    % p = succ/n;
    % xx = 0:n;
    % bp = binopdf(xx,n,p);
    % %figure;plot(xx,bp)
    % ch = ceil((1/length(condn_data1))*n);
    % %vline(ch)
    % [aa,bb] = find(xx==ch);
    % %title(num2str(sum(bp(1:bb))))
    % bino_pdf_chance(iter).n=n;
    % bino_pdf_chance(iter).succ=succ;
    % bino_pdf_chance(iter).pval=sum(bp(1:bb));
    % 
    % % binomial distribution stuff
    % n = length(test_data);
    % succ = sum(diag(acc));
    % pval = binopdf(succ,n,(1/length(condn_data1)));
    % bino_pdf(iter).n =  n;
    % bino_pdf(iter).succ = succ;
    % bino_pdf(iter).pval = pval;
    % 
    % for i=1:size(acc,1)
    %     acc1(i,:) = acc(i,:)/sum(acc(i,:));
    % end
    % for i=1:size(acc_bin,1)
    %     acc_bin1(i,:) = acc_bin(i,:)/sum(acc_bin(i,:));
    % end
    % %acc1
    % acc_permutations(iter,:,:) = acc1;
    % acc_bin_permutations(iter,:,:) = acc_bin1;


end
acc=acc_permutations;
acc_bin = [];


end

