function [V Z] = fit_bingham(X,iter,thresh)
% [V Z] = fit_bingham(X) -- fits a Bingham distribution to the sample
% vectors in the PxN sample matrix, X, with P-dimensional sample vectors in
% the columns.  Returns an orthogonal direction matrix 'V', and a diagonal
% concentration matrix, 'Z'.  By convention, Z(P,P) = 0, and Z(i,i) < 0 for
% all i < P.

if nargin < 2
   iter = 100;
end
if nargin < 3
   thresh = 1e-5;
end

P = size(X,1);
N = size(X,2);
S = X*X';  % sample matrix

% 1. Use PCA to find V and get second moments of X about V
[V D] = eig(S);
VX = V'*X;
EX2 = diag(D)./N;
EX2 = EX2(1:P-1);

% 2. Use Newton-Raphson to solve for concentrations, Z

g = @(Z) bingham_dF(Z)./bingham_F(Z) - EX2;
h = @(Z) (bingham_F(Z)*bingham_ddF(Z) - bingham_dF(Z)*bingham_dF(Z)') ./ (bingham_F(Z)^2);

z = repmat(-5, [P-1, 1])  % initialize all concentrations at -5
delta = 1;
err = inf;
for i=1:iter
   err2 = g(z);
   if norm(err2) > norm(err)  % cooling
      fprintf('****** COOLING *******\n');
      delta = .9*delta
   end
   err = err2
   if abs(err) < thresh
      break;
   end
   hz = h(z)
   dz = -hz\err;
   z = z + delta*dz

   if length(z)==2
      figure(1);
      imshow(bingham_image_2d(z(1),z(2),V(:,1),V(:,2)));
      colormap('jet');
   end
end

err = g(z);
if abs(err) >= thresh
   fprintf('Warning: fit_bingham failed to find a root in %d iterations.\n', i);
end

Z = diag([z ; 0]);
