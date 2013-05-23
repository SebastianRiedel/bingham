function p = symm_bingham_pdf(x,B)
% p = symm_bingham_pdf(x,B)

if B.Z==0  % uniform
    p = 1/B.F;
    return
end

% replicate x
X = cubic_symm(x);
k = size(X,1);

p = 0;
for i=1:k
    p = p + bingham_pdf(X(i,:), B);
end
p = p/k;
