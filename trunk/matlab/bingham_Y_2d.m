function Y = bingham_Y_2d(z1,z2,iter)
% Y = bingham_Y_2d(z1,z2,iter) - computes F/dF for the bingham
% distribution with non-zero concentrations (z1,z2), up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

F = bingham_F_2d(z1,z2,iter);
Y = bingham_dF_2d(z1,z2,iter) ./ [F ; F];
