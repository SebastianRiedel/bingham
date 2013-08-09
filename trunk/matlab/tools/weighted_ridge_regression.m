function b = weighted_ridge_regression(X, y, w, lambda)

if nargin < 4
    lambda = 0;
end

d = size(X,2);
b = ((X .* repmat(w, [1,d]))' * X + lambda*eye(d)) \ ((X .* repmat(w, [1,d]))' * y);

end
