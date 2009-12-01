function [dF G] = bingham_dF_2d(z1,z2,iter)
% dF = bingham_dF_2d(z1,z2,iter) - computes the partial derivatives of the normalization constant for the bingham
% distribution with non-zero concentrations (z1,z2), up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

log_z1 = log(abs(z1));
log_z2 = log(abs(z2));

G = zeros(iter,iter,2);
G(1,:,1) = -inf;
G(:,1,2) = -inf;

dF1 = 0;
for i=1:iter-1
    for j=0:iter-1
         g = gammaln(i+1/2) + gammaln(j+1/2) - gammaln(i+j+3/2);
         g = g + (i-1)*log_z1 + j*log_z2 - gammaln(i) - gammaln(j+1);
         G(i+1,j+1,1) = g;
         dF1 = dF1 + (-1)^(i+j-1)*exp(g);
    end
end

dF2 = 0;
for i=0:iter-1
    for j=1:iter-1
         g = gammaln(i+1/2) + gammaln(j+1/2) - gammaln(i+j+3/2);
         g = g + i*log_z1 + (j-1)*log_z2 - gammaln(i+1) - gammaln(j);
         G(i+1,j+1,2) = g;
         dF2 = dF2 + (-1)^(i+j-1)*exp(g);
    end
end

dF = [dF1 ; dF2];
