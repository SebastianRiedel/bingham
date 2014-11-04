function [M L ssd] = kmeans(X, k, num_restarts, verbose)
% [M L ssd] = kmeans(X, k, num_restarts) -- applies kmeans to the rows of sample matrix X;
% returns cluster means M (in the rows), and sample labels L

if nargin < 4
    verbose=0;
end


% kmeans with restarts
if nargin > 2
   ssd_min = inf;  M_min = [];  L_min = [];
   for i=1:num_restarts
      [M L ssd] = kmeans(X, k);
      if ssd < ssd_min
         ssd_min = ssd;
         M_min = M;
         L_min = L;
      end
   end
   ssd = ssd_min;
   M = M_min;
   L = L_min;
   return;
end


iter = 1000;


n = size(X,1);

% initialize M by randomly sampling from X (w/o replacement)
r = randperm(n);
M = X(r(1:k),:);

ssd_prev = inf;

for j=1:iter
   % update L
   D = zeros(n,k);
   for i=1:k
      MX = repmat(M(i,:), [n 1]);
      D(:,i) = sum((X - MX).*(X - MX), 2);
   end
   [dmin, L] = min(D');
   
   % update M
   for i=1:k
      M(i,:) = mean(X(find(L==i),:));
   end
   
   % compute sum of squared distances
   ssd = 0;
   for i=1:k
      Xi = X(find(L==i),:);
      Mi = repmat(M(i,:), [size(Xi,1) 1]);
      ssd = ssd + sum(sum((Xi-Mi).*(Xi-Mi)));
   end
   %fprintf('%d. ssd = %f\n', j, ssd);
   if verbose, if mod(j,10)==0, fprintf('.'); end, end
   
   if abs(ssd - ssd_prev) < 1e-4
      break;
   end
   
   ssd_prev = ssd;
end

