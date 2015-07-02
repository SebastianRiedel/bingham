function [B outliers] = bingham_fit_mlesac(X)
% [B outliers] = bingham_fit_mlesac(X) -- where B = B.{V, Z, F}
X = X';
d = size(X,1)
n = size(X,2)

iter = 100;
p0 = 1 / surface_area_hypersphere(d-1);  % uniform density for outliers
logp0 = log(p0);

pmax = -inf;

for i=1:iter

   % pick d points at random from X
   r = randperm(n);
   r = r(1:d)
   for j=1:d
        eval(['X_' num2str(j) '= X(:,r(j));'])
%    Xi = X(:,r);
   end   
   % fit a Bingham to the d points
   X_combined = [X_1 X_2 X_3 X_4];
   X_combined = X_combined';
   bing_X_combined = bingham_fit(X_combined);
   %[V Z F] = bingham_fit_scatter(Xi*Xi')
   V = bing_X_combined.V;
   Z = bing_X_combined.Z;
   F = bing_X_combined.F;
   % compute data log likelihood
   logp = 0;
  
   for j=1:n
      p = bingham_pdf(X(:,j)', bing_X_combined);
      if p > p0
         logp = logp + log(p);
      else
         logp = logp + logp0;
      end
   end
   
   if logp > pmax
      pmax = logp;
      bing_X_combined.V = V;
      bing_X_combined.Z = Z;
      bing_X_combined.F = F;
      
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
   p = bingham_pdf(X(:,j), bing_X_combined);
   if p > p0
      L(j) = 1;
   else
      L(j) = 0;
   end
end

inliers = find(L);
outliers = find(~L);

bing_return = bingham_fit(X(:,inliers)');
B = bing_return; 



