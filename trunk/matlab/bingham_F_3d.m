function F = bingham_F_3d(z1,z2,z3,iter)
% F = bingham_F_3d(z1,z2,z3,iter) - computes the normalization constant for the bingham
% distribution with non-zero, negative concentrations (z1,z2,z3), up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

log_z1 = log(abs(z1));
log_z2 = log(abs(z2));
log_z3 = log(abs(z3));

F = 0;
for i=0:iter-1
    for j=0:iter-1
        for k=0:iter-1
            g = gammaln(i+1/2) + gammaln(j+1/2) + gammaln(k+1/2) - gammaln(i+j+k+2);
            g = g + i*log_z1 + j*log_z2 + k*log_z3 - gammaln(i+1) - gammaln(j+1) - gammaln(k+1);
            F = F + (-1)^(i+j+k)*exp(g);
        end
    end
end
