function y = interp_linear(X,alpha)
%y = interp_linear(X,alpha) -- interpolate between the 2x2x...x2 N-D matrix X
%with mixing coefficients alpha.

s = size(X);
if s(end)==1
    s = s(1:end-1);
end
d = length(s);

Z = X(1:end); %reshape(X, [1,2^d]);
if d==3  % optimization for typical case
    Z = Z(1:4) + alpha(3)*(Z(5:8) - Z(1:4));
    Z = Z(1:2) + alpha(2)*(Z(3:4) - Z(1:2));
    y = Z(1) + alpha(1)*(Z(2) - Z(1));
else
    for i=d:-1:1
        k = 2^(i-1); %length(Z)/2;
        Z = Z(1:k) + alpha(i)*(Z(k+1:end) - Z(1:k));
    end
    y = Z;
end
