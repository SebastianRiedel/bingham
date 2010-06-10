function X = bingham_mixture_sample(B,W,n)
% X = bingham_mixture_sample(B,W,n) - sample n points from a Bingham
% mixture using Monte Carlo simulation

X = [];
for i=1:length(B)
    X = [X ; bingham_sample(B(i),ceil(W(i)*n))];
end
