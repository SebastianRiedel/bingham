function i = pmf_sample(W)
%i = pmf_sample(W) -- draw a sample from a discrete probability
%distribution

if size(W,1)>1
    W = W';
end

W = W/sum(W);  % normalize

r = rand();
C = cumsum(W);
i = find(([0 C(1:end-1)]<r).*(C>=r));
