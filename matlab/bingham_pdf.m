function p = bingham_pdf(x,B)
% p = bingham_pdf(x,B)
% 
% INPUTS:
% x   should be N-by-D
% B.V should be D-by-(D-1)
% B.Z should be (D-1)-by-1 or 1-by-(D-1)
%
% OUTPUT:
% p   is N-by-1

if B.Z==0  % uniform
    p = 1/B.F;
    return
end

% make z a column vector
z = B.Z(:);

p = exp((x*B.V).^2*z) / B.F;
