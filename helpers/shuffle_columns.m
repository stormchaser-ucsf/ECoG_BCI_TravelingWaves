function Y = shuffle_columns(X)
[N, P] = size(X);
Y = zeros(N, P, 'like', X);
for p = 1:P
    Y(:, p) = X(randperm(N),p);  % accurate independent column shuffle
end
end