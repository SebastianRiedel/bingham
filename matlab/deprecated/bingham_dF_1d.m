function dF = bingham_dF_1d(z1,iter)
% dF = bingham_dF_1d(z1,iter) - computes the derivative of the normalization constant for the bingham
% distribution with non-zero concentration 'z1', up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

log_z1 = log(abs(z1));

dF1 = 0;
for i=1:iter-1
   g = gammaln(i+1/2) - gammaln(i+1);
   g = g + (i-1)*log_z1 - gammaln(i);
   dF1 = dF1 + (-1)^(i-1)*exp(g);
end

dF = dF1;
