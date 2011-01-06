function X = bingham_mixture_sample(B,W,n)
% X = bingham_mixture_sample(B,W,n) - sample n points from a Bingham
% mixture using Monte Carlo simulation


if n==1
    
    % sample a mixture component
    i = pmf_sample(W);
    X = bingham_sample(B(i), 1);
    
elseif n<100
    
    for i=1:n
        X(i,:) = bingham_mixture_sample(B,W,1);
    end
    
else  % n >= 100

    X = [];
    for i=1:length(B)
        X = [X ; bingham_sample(B(i),round(W(i)*2*n))];
        p = randperm(size(X,1));
        X = X(p(1:n),:);
    end
end