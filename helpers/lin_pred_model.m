function recon_error = lin_pred_model(x,y)


train_idx = randperm(length(x),round(0.8*length(x)));
I =  ones(length(x),1);
I(train_idx)=0;
test_idx = find(I==1);

xtrain = x(train_idx);
xtrain = cell2mat(xtrain);
xtrain = xtrain(:,:);
%xtrain=real(xtrain);

ytrain = y(train_idx);
ytrain = cell2mat(ytrain);
ytrain = ytrain(:,:);
%ytrain=real(ytrain);

% build the regression model
A = pinv(xtrain)*ytrain;

% test on held out samples
xtest=x(test_idx);
ytest=y(test_idx);

% for i=1:length(xtest)
%     tmp = xtest{i};
%     tmp = tmp(:,:);
% 
%     tmpy = ytest{i};
%     tmpy = tmpy(:,:);
% 
%     predy = tmp*A;
% end


ytest = cell2mat(ytest);
ytest = ytest(:,:);
%ytest=real(ytest);

xtest = cell2mat(xtest);
xtest = xtest(:,:);
%xtest=real(xtest);

predy = xtest*A;

recon_error = ytest - predy;
recon_error = norm(recon_error,'fro')/size(recon_error,1);


end