function B = bingham_fit_ransac(X)
% B = bingham_fit_ransac(X) -- where B = B.{V, Z, F}

d = size(X,1);
n = size(X,2);

iter = 100;
p0 = 1 / surface_area_hypersphere(d-1);  % uniform density for outliers

max_inliers = 0;

for i=1:iter

   % pick d points at random from X
   r = randperm(n);
   r = r(1:d);
   Xi = X(:,r);
   
   % fit a Bingham to the d points
   [V Z F] = bingham_fit(Xi);
   %[V Z F] = bingham_fit_scatter(Xi*Xi')
   
   % count inliers
   cnt = 0;
   for j=1:n
      if bingham_pdf(X(:,j), V, Z, F) > p0
         cnt = cnt+1;
      end
   end
   
   if cnt > max_inliers
      max_inliers = cnt;
      B.V = V;
      B.Z = Z;
      B.F = F;
      
      %fprintf('*** found new best with %d/%d inliers ***\n', cnt, n);
      
      %figure(20);
      %plot_bingham_3d(V,Z,F,X');
      %figure(21);
      %plot_bingham_3d_projections(V,Z,F);
      
   end
      
end
   
   