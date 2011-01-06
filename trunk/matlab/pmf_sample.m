function i = pmf_sample(W)
%i = pmf_sample(W) -- draw a sample from a discrete probability
%distribution

r = rand();
C = cumsum(W);
i = find(([0 C(1:end-1)]<r).*(C>=r));
