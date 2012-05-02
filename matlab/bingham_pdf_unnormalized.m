function p = bingham_pdf_unnormalized(x,B)
% p = bingham_pdf_unnormalized(x,B)

% make x a row vector
if size(x,1)>1
    x = x';
end

% make z a row vector
z = B.Z;
if size(z,1)>1
    z = z';
end

p = exp(sum(z.*(x*B.V).^2));
