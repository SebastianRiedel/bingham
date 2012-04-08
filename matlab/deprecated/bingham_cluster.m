function [B cnts] = bingham_cluster(X, min_points)
% [B cnts] = bingham_cluster(X, min_points) -- where each B(i) contains fields V, Z, F

if nargin < 2
   min_points = 20;
end

for i=1:100

   [B(i) outliers] = bingham_fit_mlesac(X);
   cnts(i) = size(X,2) - length(outliers);
   
   if length(outliers) < min_points
      break;
   end
   
   X = X(:, outliers);
end
