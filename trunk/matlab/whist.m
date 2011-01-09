function H = whist(Y, W, X)
%H = whist(Y, W, X) -- weighted histogram with bin centers 'X', data 'Y'
%and weights 'W'

E = [(X(2:end) + X(1:end-1))/2, inf];

H = zeros(size(E));
for i=1:length(Y)
    bin = find(Y(i)<E, 1);
    H(bin) = H(bin) + W(i);
end

