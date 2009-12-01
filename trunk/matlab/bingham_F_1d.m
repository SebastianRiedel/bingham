function F = bingham_F_1d(z1,iter)
% F = bingham_F_2d(z1,iter) - computes the normalization constant for the bingham
% distribution with non-zero concentration 'z1', up to a given number of terms
% (per dimension) in the infinite series, 'iter'.


log_z1 = log(abs(z1));

F = 0;
for i=0:iter-1
   g = gammaln(i+1/2) - gammaln(i+1);
   g = g + i*log_z1 - gammaln(i+1);
   F = F + (-1)^(i)*exp(g);
end
