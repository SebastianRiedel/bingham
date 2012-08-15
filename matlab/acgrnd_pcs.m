function X = acgrnd_pcs(z,V,n)
%X = acgrnd_pcs(z,V,n) -- z and V are the sqrt(eigenvalues) and
%eigenvectors of the covariance matrix

if size(z,1)>1
    z = z';
end

if nargin < 3 || n == 1
    d = length(z);
    x = (randn(1,d).*z)*V';
    X = x/norm(x);
else
    d = length(z);
    X = (randn(n,d).*repmat(z,[n,1]))*V';
    X = X ./ repmat(sqrt(sum(X.^2,2)), [1 d]);
end
