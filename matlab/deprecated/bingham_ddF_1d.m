function H = bingham_ddF_1d(z1,iter)
% H = bingham_ddF_1d(z1,iter) - computes the second derivative of the normalization constant for the bingham
% distribution with non-zero concentration 'z1', up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

log_z1 = log(abs(z1));

ddF = 0;
for i=2:iter-1
   g = gammaln(i+1/2) - gammaln(i+1);
   g = g + (i-2)*log_z1 - gammaln(i-1);
   ddF = ddF + (-1)^(i-2)*exp(g);
end

H = ddF;
