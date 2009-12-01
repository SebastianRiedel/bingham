function F = bingham_F_2d(z1,z2,iter)
% F = bingham_F_2d(z1,z2,iter) - computes the normalization constant for the bingham
% distribution with non-zero concentrations (z1,z2), up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

log_z1 = log(abs(z1));
log_z2 = log(abs(z2));

F = 0;
for i=0:iter-1
    for j=0:iter-1
         g = gammaln(i+1/2) + gammaln(j+1/2) - gammaln(i+j+3/2);
         g = g + i*log_z1 + j*log_z2 - gammaln(i+1) - gammaln(j+1);
         F = F + (-1)^(i+j)*exp(g);
    end
end
