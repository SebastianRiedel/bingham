function E = tofoo_error_rotsym(fdir, files, bag_sizes)
% E = tofoo_error_rotsym(fdir, files)

% read in the indices
I = [];
J = [];
K = [];
for i=1:length(files)
   a = sscanf(files(i).name, 'cv%d_post%d_bag%d.m');
   files(i).name
   %a = sscanf(files(i).name, 'cv%d_post%d_nxyz_fpfh_bag%d.m');
   K = [K a(1)];
   I = [I a(2)];
   J = [J a(3)];
end
K = intersect(K, 1:max(K));
%I = intersect(I, 1:max(I));
J = intersect(J, 1:max(J));

if nargin > 2
   J = intersect(J, bag_sizes);
end

if length(K) > 1
   for k=1:length(K)
      fk = [];
      for i=1:length(files)
         if ~isempty(strfind(files(i).name, sprintf('cv%d_', K(k))))
            fk = [fk files(i)];
         end
      end
      E(:,:,:,K(k)) = tofoo_error_rotsym(fdir, fk);
   end
   return;
end

% E(:, bag_size, pcd, cv)
E = zeros(100, length(J), length(I), length(K));
for k=1:length(K)
   for i=1:length(I)
      for j = 1:length(J)
         run(sprintf('%s/cv%d_post%d_bag%d.m', fdir, K(k), I(i), J(j)));
         %run(sprintf('%s/cv%d_post%d_nxyz_fpfh_bag%d.m', fdir, K(k), I(i), J(j)));
         theta = acos(abs(X(:,1).^2 - X(:,2).^2 - X(:,3).^2 + X(:,4).^2))';
         theta = (180/pi)*theta;
         if length(theta) > 100
            theta = theta(1:100);
         end
         n = length(theta);
         if n < 100
            theta = [theta repmat(min(theta), [1 100-n])];
         end
         for t=1:100
            theta(t) = min(theta(1:t));
         end
         E(:,j,i,k) = theta;
      end
   end
end




