function dF = bingham_dF_3d(z1,z2,z3,iter)
% dF = bingham_dF_3d(z1,z2,z3,iter) - computes the partial derivatives of the normalization constant for the bingham
% distribution with non-zero concentrations (z1,z2,z3), up to a given number of terms
% (per dimension) in the infinite series, 'iter'.


log_z1 = log(abs(z1));
log_z2 = log(abs(z2));
log_z3 = log(abs(z3));

dF1 = 0;
for i=1:iter-1
    for j=0:iter-1
        for k=0:iter-1
            g = gammaln(i+1/2) + gammaln(j+1/2) + gammaln(k+1/2) - gammaln(i+j+k+2);
            g = g + (i-1)*log_z1 + j*log_z2 + k*log_z3 - gammaln(i) - gammaln(j+1) - gammaln(k+1);
            dF1 = dF1 + (-1)^(i+j+k-1)*exp(g);
        end
    end
end

dF2 = 0;
for i=0:iter-1
    for j=1:iter-1
        for k=0:iter-1
            g = gammaln(i+1/2) + gammaln(j+1/2) + gammaln(k+1/2) - gammaln(i+j+k+2);
            g = g + i*log_z1 + (j-1)*log_z2 + k*log_z3 - gammaln(i+1) - gammaln(j) - gammaln(k+1);
            dF2 = dF2 + (-1)^(i+j+k-1)*exp(g);
        end
    end
end

dF3 = 0;
for i=0:iter-1
    for j=0:iter-1
        for k=1:iter-1
            g = gammaln(i+1/2) + gammaln(j+1/2) + gammaln(k+1/2) - gammaln(i+j+k+2);
            g = g + i*log_z1 + j*log_z2 + (k-1)*log_z3 - gammaln(i+1) - gammaln(j+1) - gammaln(k);
            dF3 = dF3 + (-1)^(i+j+k-1)*exp(g);
        end
    end
end

dF = [dF1 ; dF2 ; dF3];
