function p = acgpdf_pcs(x,z,V)
%p = acgpdf_pcs(x,z,V) -- z and V are the sqrt(eigenvalues) and
%eigenvectors of the covariance matrix

% make x a row vector
if size(x,1)>1
    x = x';
end

S_inv = V*diag(1./(z.^2))*V';

d = length(x);
p = 1 / (prod(z) * surface_area_hypersphere(d-1));
md = x*S_inv*x';  % mahalanobis distance
p = p * md^(-d/2);
