function [M L] = kmeans_chi2(X, k)
% [M L] = kmeans_chi2(X, k) -- applies kmeans (with a chi^2 distance) to the
% rows of sample matrix X;
% returns cluster means M (in the rows), and sample labels L


iter = 1000;


n = size(X,1);

% initialize M by randomly sampling from X (w/o replacement)
r = randperm(n);
M = X(r(1:k),:);

ssd_prev = inf;

for cnt=1:iter
   % update L
   D = zeros(n,k);
   for i=1:k
      MX = repmat(M(i,:), [n 1]);
      D(:,i) = sum((X - MX).*(X - MX)./(X + MX), 2);
   end
   [dmin, L] = min(D');
   
   % update M
   M_cnt = 0;
   for i=1:k
      for j=1:size(X,2)
         % use Newton-Raphson to compute M(i,j)
         X_ij = X(find(L==i),j);
         n_ij = length(X_ij);
         M_ij = mean(X_ij);
         
         %fprintf('X_ij = [');
         %fprintf('%f, ', X_ij');
         %fprintf(']\n');
         
         if sum(L==i) == 0  % cluster is empty --> assign it to furthest outlier
            %fprintf('*** Cluster is empty! ***\n');
            [dmax imax] = max(dmin);
            M(i,:) = X(imax,:);
         end
         
         if M_ij < eps
            M(i,j) = M_ij;
            continue;
         end
         
         % dbug
%          u = 0:.01*M_ij:2*M_ij;
%          x = X_ij;
%          g = zeros(size(u));
%          for ui=1:length(u)
%             g(ui) = sum( (x-u(ui)).*(x-u(ui))./(u(ui)+x) );
%          end
%          figure(10);
%          plot(u, g);
%          hold on;
%          plot([u(1) u(end)], [0 0], 'k-');
%          hold off;
         
         while 1
            M_ij_prev = M_ij;
            M_cnt = M_cnt + 1;
            f = n_ij/4 - sum( (X_ij./(M_ij + X_ij)).^2 );
            df = 2*sum( X_ij.^2 ./ (M_ij + X_ij).^3 );
            %fprintf('.');
            %fprintf('M(%d,%d) = %f, f = %f, df = %f, f/df = %f\n', i, j, M_ij, f, df, f/df);
            %pause(.01);
            M_ij = max(M_ij - f/df, .001);
            %if abs(f/df) < max((.0001)*M_ij, eps)
            if abs(M_ij - M_ij_prev) < max((.0001)*M_ij, eps)
               break;
            end
         end
         %hold on;
         %plot(M_ij, sum( (x-M_ij).*(x-M_ij)./(M_ij+x) ), 'ro', 'LineWidth', 2);
         %hold off;
         %fprintf('\n');
         %pause(.5);
         M(i,j) = M_ij;
      end
   end
   
   % compute sum of squared distances
   ssd = 0;
   for i=1:k
      Xi = X(find(L==i),:);
      Mi = repmat(M(i,:), [size(Xi,1) 1]);
      ssd = ssd + sum(sum((Xi-Mi).*(Xi-Mi)./(Xi + Mi)));
   end
   fprintf('%d. ssd = %f, M avg iter = %.2f\n', cnt, ssd, M_cnt / (k*size(X,2)));
   
   if abs(ssd - ssd_prev) < 1e-4
      break;
   end
   
   ssd_prev = ssd;
end

