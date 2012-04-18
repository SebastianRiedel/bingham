function p = acgpdf(x,S)
%p = acgpdf(x,S)

% make x a row vector
if size(x,1)>1
    x = x';
end

d = length(x);
p = 1 / (sqrt(det(S)) * surface_area_hypersphere(d-1));
md = x*inv(S)*x';  % mahalanobis distance
p = p * md^(-d/2);
