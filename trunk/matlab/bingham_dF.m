function dF = bingham_dF(z,iter)
% dF = bingham_dF(z,iter) - computes the gradient of the normalization constant for the bingham
% distribution with non-zero concentration vector 'z', up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

if nargin < 2
   iter = 80;
end

if length(z)==1
   dF = bingham_dF_1d(z, iter);
elseif length(z)==2
   dF = bingham_dF_2d(z(1), z(2), iter);
elseif length(z)==3
   dF = bingham_dF_3d(z(1), z(2), z(3), iter);
else
   fprintf('Error: bingham_dF() currently supports only 1-3 dimensions');
   dF = [];
end
