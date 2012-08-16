function P = acgpdf_pcs(X,z,V)
%P = acgpdf_pcs(X,z,V) -- z and V are the sqrt(eigenvalues) and
%eigenvectors of the covariance matrix; x's are in the rows of X

% make x a row vector
%if size(x,1)>1
%    x = x';
%end

S_inv = V*diag(1./(z.^2))*V';

d = size(X,2);
P = repmat(1 / (prod(z) * surface_area_hypersphere(d-1)), [size(X,1),1]);
md = sum((X*S_inv).*X, 2);  % mahalanobis distance
P = P .* md.^(-d/2);
