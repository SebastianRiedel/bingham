function p = bingham_pdf(x,B)
% p = bingham_pdf(x,B)

d = length(x);

if B.Z==0  % uniform
    p = 1/B.F;
    return
end

% make x a row vector
if size(x,1)>1
    x = x';
end

% make z a row vector
z = B.Z;
if size(z,1)>1
    z = z';
end

p = exp(sum(z.*(x*B.V).^2)) / B.F;
