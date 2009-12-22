function [M L b] = kmeans_bic(X)
% [M L] = kmeans(X) -- applies kmeans to the rows of sample matrix X and uses
% the Bayesian Information Criterion (BIC) to determine the number of clusters, k;
% returns cluster means M (in the rows), and sample labels L

num_restarts = 5;

n = size(X,1);
p = size(X,2);

b = [];
bic_min = inf;  M_min = [];  L_min = [];
for k=2:sqrt(n)/3
   [M L ssd] = kmeans(X, k, num_restarts);
   bic = n*(1 + log(ssd)) + (k*p-n)*log(n);
   b(k) = bic;
   fprintf('\n(k=%d)  bic = %f', k, bic);
   if bic < bic_min
      bic_min = bic;
      M_min = M;
      L_min = L;
   end
end
fprintf('\n');

M = M_min;
L = L_min;
