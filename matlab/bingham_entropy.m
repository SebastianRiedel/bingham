function h = bingham_entropy(B)
% h = bingham_entropy(B)

F = B.F;
dF = B.dF;
Z = B.Z;

h = log(F) - Z'*dF/F;
