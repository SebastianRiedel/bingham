function [B outliers] = bingham_fit_mlesac(X)
% [B outliers] = bingham_fit_mlesac(X) -- where B = B.{V, Z, F}
X = X';
d = size(X,1);
n = size(X,2);

iter = 100;
p0 = 1 / surface_area_hypersphere(d-1);  % uniform density for outliers
logp0 = log(p0);

pmax = -inf;

for i=1:iter
   fprintf('mlesac iteration: %d', i);

   % pick d points at random from X -> put them into X_1, X_2, X_3, X_4
   r = randperm(n);
   r = r(1:d);
   for j=1:d
        eval(['X_' num2str(j) '= X(:,r(j));'])
   end   
   X_combined = [X_1 X_2 X_3 X_4];
   X_combined = X_combined';

   % fit a Bingham to the d points
   bing_X_combined = bingham_fit(X_combined);

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
   
   % update the threshold
   if logp > pmax
      pmax = logp;
   end

end

% find inliers/outliers
L = zeros(1,n);
for j=1:n
   p = bingham_pdf(X(:,j)', bing_X_combined);
   if p > p0
      L(j) = 1;
   else
      L(j) = 0;
   end
end

inliers = find(L);
outliers = find(~L);
fprintf('no of outliers were %d', outliers);

% fit a Bingham to all the inliers
bing_return = bingham_fit(X(:,inliers)');
B = bing_return; 



