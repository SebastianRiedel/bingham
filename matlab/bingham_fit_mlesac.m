function [B outliers] = bingham_fit_mlesac(X)
% [B outliers] = bingham_fit_mlesac(X) -- where B = B.{V, Z, F}

d = size(X,1);
n = size(X,2);

iter = 100;
p0 = 1 / surface_area_hypersphere(d-1);  % uniform density for outliers
logp0 = log(p0);

pmax = -inf;

for i=1:iter

   % pick d points at random from X
   r = randperm(n);
   r = r(1:d);
   Xi = X(:,r);
   
   % fit a Bingham to the d points
   [V Z F] = bingham_fit(Xi);
   %[V Z F] = bingham_fit_scatter(Xi*Xi')
   
   % compute data log likelihood
   logp = 0;
   for j=1:n
      p = bingham_pdf(X(:,j), V, Z, F);
      if p > p0
         logp = logp + log(p);
      else
         logp = logp + logp0;
      end
   end
   
   if logp > pmax
      pmax = logp;
      B.V = V;
      B.Z = Z;
      B.F = F;
      
      %fprintf('*** found new best with log likelihood %f ***\n', logp);
      
      %figure(20);
      %plot_bingham_3d(V,Z,F,X');
      %figure(21);
      %plot_bingham_3d_projections(V,Z,F);
      
   end
end

% find inliers/outliers and fit the Bingham to all the inliers

L = zeros(1,n);
for j=1:n
   p = bingham_pdf(X(:,j), B.V, B.Z, B.F);
   if p > p0
      L(j) = 1;
   else
      L(j) = 0;
   end
end

inliers = find(L);
outliers = find(~L);

[B.V B.Z B.F] = bingham_fit(X(:,inliers));





