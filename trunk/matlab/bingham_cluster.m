function B = bingham_cluster(X)
% B = bingham_cluster(X) -- where each B(i) contains fields V, Z, F

min_points = 10;

for i=1:100

   [B(i) outliers] = bingham_fit_mlesac(X);

   if length(outliers) < min_points
      break;
   end
   
   X = X(:, outliers);
end
