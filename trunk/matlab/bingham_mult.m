function B = bingham_mult(B1, B2, normalize)
%B = bingham_mult(B1, B2) -- Multiply two Bingham PDFs.
%B = bingham_mult(B1, B2, normalize) -- Multiply two Bingham PDFs.
%B = bingham_mult(Bs) -- Multiply a vector of Bingham PDFs.
%B = bingham_mult(Bs, normalize) -- Multiply a vector of Bingham PDFs.

bingham_min_concentration = -900;  % TODO: move this somewhere else

if nargin < 2 || ~isstruct(B2)
    n = length(B1);
    B.d = B1(1).d;
    C = zeros(B.d);
    for i=1:n
        C = C + B1(i).V * diag(B1(i).Z) * B1(i).V';
    end
    if nargin==2
        normalize = B2;
    else
        normalize = 1;
    end
else
    B.d = B1.d;
    C = B1.V * diag(B1.Z) * B1.V' + B2.V * diag(B2.Z) * B2.V';
    if nargin < 3
        normalize = 1;
    end
end

%if isfinite(C)
% eigenvalues will be sorted from largest to smallest in magnitude
C = (C+C')/2;  % fix numerical asymmetries
[V D] = eig(C);
z = diag(D)';

% set the smallest z (in magnitude) to zero
B.V = V(:,1:3);
B.Z = max(z(1:3) - z(4), bingham_min_concentration);

% lookup constants
if normalize
    [F dF] = bingham_F(B.Z);
    B.F = F;
    B.dF = dF;
end
%else
%    fprintf('Error (bingham_mult): C is not finite!\n');
%end