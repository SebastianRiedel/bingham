function H = bingham_ddF_2d(z1,z2,iter)
% H = bingham_ddF_2d(z1,z2,iter) - computes the Hessian matrix of second partial
% derivatives of the normalization constant for the bingham distribution with
% respect to non-zero concentrations (z1,z2), up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

log_z1 = log(abs(z1));
log_z2 = log(abs(z2));

h11 = 0;
for i=2:iter-1
    for j=0:iter-1
         g = gammaln(i+1/2) + gammaln(j+1/2) - gammaln(i+j+3/2);
         g = g + (i-2)*log_z1 + j*log_z2 - gammaln(i-1) - gammaln(j+1);
         h11 = h11 + (-1)^(i+j-2)*exp(g);
    end
end

h12 = 0;
for i=1:iter-1
    for j=1:iter-1
         g = gammaln(i+1/2) + gammaln(j+1/2) - gammaln(i+j+3/2);
         g = g + (i-1)*log_z1 + (j-1)*log_z2 - gammaln(i) - gammaln(j);
         h12 = h12 + (-1)^(i+j-2)*exp(g);
    end
end

h22 = 0;
for i=0:iter-1
    for j=2:iter-1
         g = gammaln(i+1/2) + gammaln(j+1/2) - gammaln(i+j+3/2);
         g = g + i*log_z1 + (j-2)*log_z2 - gammaln(i+1) - gammaln(j-1);
         h22 = h22 + (-1)^(i+j-2)*exp(g);
    end
end

H = [h11 h12 ; h12 h22];
