function H = bingham_ddF(z,iter)
% H = bingham_ddF(z,iter) - computes the Hessian matrix of second partial
% derivatives of the normalization constant for the bingham distribution with
% respect to non-zero concentrations 'z', up to a given number of terms
% (per dimension) in the infinite series, 'iter'.

if nargin < 2
   iter = 80;
end

if length(z)==1
   H = bingham_ddF_1d(z, iter);
elseif length(z)==2
   H = bingham_ddF_2d(z(1), z(2), iter);
elseif length(z)==3
   H = bingham_ddF_3d(z(1), z(2), z(3), iter);
else
   fprintf('Error: bingham_ddF() currently supports only 1-3 dimensions');
   H = [];
end
