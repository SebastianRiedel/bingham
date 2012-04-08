function p = occ_grid_pdf(X, occ_grid, lambda)
%p = occ_grid_pdf(X, occ_grid, lambda)

if nargin < 3
    lambda = 1;
end

n = size(X,1);
occ_logp = 0;
for i=1:n
    x = X(i,:);
    c = ceil((x - occ_grid.min)/occ_grid.res);    % point cell
    if (c >= [1,1,1]) .* (c <= size(occ_grid.occ))
        logp = log(occ_grid.occ(c(1), c(2), c(3)));
    else
        logp = log(.5);
    end
    occ_logp = occ_logp + logp;
end
p = lambda*exp(lambda*occ_logp/n);

