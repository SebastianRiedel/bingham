function X = acgrnd(S,n)
%X = acgrnd(S,n)

if nargin < 2 || n == 1
    d = size(S,1);
    x = mvnrnd(zeros(1,d), S);
    X = x/norm(x);
else
    d = size(S,1);
    X = mvnrnd(zeros(1,d), S, n);
    X = X ./ repmat(sqrt(sum(X.^2,2)), [1 d]);
end
