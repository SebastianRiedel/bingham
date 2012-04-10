function y = interp_linear(X,alpha)
%y = interp_linear(X,alpha) -- interpolate between the 2x2x...x2 N-D matrix X
%with mixing coefficients alpha.

d = length(size(X));
Z = reshape(X, [1,2^d]);
for i=d:-1:1
    k = length(Z)/2;
    Z = Z(1:k) + alpha(i)*(Z(k+1:end) - Z(1:k));
end
y = Z;
