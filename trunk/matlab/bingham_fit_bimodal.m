function [V1 Z1 F1 V2 Z2 F2] = bingham_fit_bimodal(X)
% [V1 Z1 F1 V2 Z2 F2] = bingham_fit_bimodal(X)

iter = 20;
z0 = -5;

d = size(X,1);
n = size(X,2);

% initialize binghams
I = eye(d);
V1 = I(:,1:d-1);
V2 = I(:,2:d);
Z1 = repmat(z0, [d-1 1]);
Z2 = repmat(z0, [d-1 1]);
F1 = 1;
F2 = 1;

L = zeros(1,n);

for cnt=1:iter
   
   if d==4
      figure(11);
      plot_bingham_3d_projections(V1, Z1, F1);
      figure(12);
      plot_bingham_3d_projections(V2, Z2, F2);
      fprintf('.');
   end

   % compute labels
   for i=1:n
      if bingham_pdf(X(:,i), V1, Z1, F1) > bingham_pdf(X(:,i), V2, Z2, F2)
         L(i) = 1;
      else
         L(i) = 2;
      end
   end
   
   % fit binghams
   [V1 Z1 F1] = bingham_fit(X(:,L==1));
   [V2 Z2 F2] = bingham_fit(X(:,L==2));
   
end
   
   