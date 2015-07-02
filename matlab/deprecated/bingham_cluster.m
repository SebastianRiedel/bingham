function [B cnts] = bingham_cluster(X, min_points)
% [B cnts] = bingham_cluster(X, min_points) -- where each B(i) contains fields V, Z, F

if nargin < 2
   min_points = 20;
end

for i=1:6

   [B(i) outliers] = bingham_fit_mlesac(X);
%    outliers
%   ouliers is a row vector
%    size(X,2)
%    i
   cnts(i) = size(X,1) - length(outliers);
   
   if length(outliers) < min_points
      break;
   end
  X_updater = zeros(length(outliers), size(X,2));
%    size(X,1)
%    size(X,2)
  for j = 1:length(outliers)
    X_updater(j,:) = X(outliers(j),:);
  end
  X = X_updater;
  size(X)
end
