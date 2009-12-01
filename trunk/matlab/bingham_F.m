function F = bingham_F(z,iter)
% F = bingham_F(z,iter) - computes the normalization constant for the bingham
% distribution with non-zero concentration vector 'z', up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

if nargin < 2
   iter = 80;
end

if length(z)==1
   F = bingham_F_1d(z, iter);
elseif length(z)==2
   F = bingham_F_2d(z(1), z(2), iter);
elseif length(z)==3
   F = bingham_F_3d(z(1), z(2), z(3), iter);
else
   fprintf('Error: bingham_F() currently supports only 1-3 dimensions');
   F = [];
end
