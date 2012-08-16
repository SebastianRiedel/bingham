function P = bingham_pdf_unnormalized(X,B)
% P = bingham_pdf_unnormalized(X,B) -- x's in the rows

% make x a row vector
%if size(x,2)>1
%    x = x';
%end

% make z a row vector
z = B.Z;
if size(z,1)>1
    z = z';
end

Z = repmat(z,[size(X,1),1]);

P = exp(sum(Z.*(X*B.V).^2, 2));
