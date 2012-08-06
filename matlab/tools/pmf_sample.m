function I = pmf_sample(W,n)
%I = pmf_sample(W,n) -- draw n samples from a discrete probability
%distribution

if nargin < 2
    n = 1;
end

if size(W,1)>1
    W = W';
end

W = W/sum(W);  % normalize
C = cumsum(W);

r = rand(1,n);
I = zeros(1,n);
for i=1:n
    I(i) = find(([0 C(1:end-1)]<r(i)).*(C>=r(i)));
end
