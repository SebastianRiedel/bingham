function B = bingham_mult(B1, B2)
%B = bingham_mult(B1, B2) -- Multiply two Bingham PDFs.

bingham_min_concentration = -900;  % TODO: move this somewhere else

if B1.d ~= B2.d
    fprintf('Error: B1.d != B2.d in bingham_mult()!\n');
    return;
end
B.d = B1.d;

C1 = B1.V * diag(B1.Z) * B1.V';
C2 = B2.V * diag(B2.Z) * B2.V';

% eigenvalues will be sorted from largest to smallest in magnitude
[V D] = eigs(C1+C2);
z = diag(D)';

% set the smallest z (in magnitude) to zero
B.V = V(:,1:3);
B.Z = max(z(1:3) - z(4), bingham_min_concentration);

% lookup constants
[F dF] = bingham_F(B.Z);
B.F = F;
B.dF = dF;
