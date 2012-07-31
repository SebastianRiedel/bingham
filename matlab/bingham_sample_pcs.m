function X = bingham_sample_pcs(B, n, prob_thresh)
% X = bingham_sample_pcs(B, n, prob_thresh)

Z = B.Z;
V = B.V;
F = B.F;
d = length(Z);

if max(Z) == 0  %dbug--fix this case!
    X = [];
    return
end

if nargin < 3
   prob_thresh = 1/surface_area_hypersphere(d);
end
p0 = prob_thresh;

Y = 1./sqrt(-Z);
m = exp((log(n) - sum(log(Y)))/d);
pcs_grid_size = ceil(m*Y);  % num. samples in each pc direction
pcs_grid_size = 2*ceil(pcs_grid_size/2)-1;  % round to the nearest odd number
cmax = sqrt(log(F*p0)./Z);
cmax = min(cmax, 1);
%theta_min = acos(cmax)
%theta_step = 2*(pi/2 - theta_min) ./ (pcs_grid_size-1) - eps
step = 2*cmax ./ (pcs_grid_size-1) - eps;

grid_size = round(exp(sum(log(pcs_grid_size))));
X = zeros(grid_size, d+1);

for i=1:d
   if pcs_grid_size(i) == 1
      c = 0;
   else
      %theta1 = (pi/2):-theta_step(i):theta_min(i)
      %c1 = cos(theta1)  %dbug
      c1 = 0:step(i):cmax(i);
      c = [-c1(end:-1:2) c1];
   end
   pcs_grid_coords{i} = c;
end

if d==3

   % search for axis to cut the hypersphere in half
   det_V = [det(V([2 3 4],:)), det(V([1 3 4],:)), det(V([1 2 4],:)), det(V([1 2 3],:))];
   [dmax imax] = max(abs(det_V));
   cut_axis = imax;
   
   uncut_axes = [1:cut_axis-1, cut_axis+1:4];
   c0 = V(cut_axis,:)';
   W = inv(V(uncut_axes,:))';
   
   x = zeros(1,d+1);
   x(cut_axis) = 1;
   cnt = 1;
   for i=1:pcs_grid_size(1)
      for j=1:pcs_grid_size(2)
         for k=1:pcs_grid_size(3)

            % sample uncut coords x = W*(c-c0) and normalize
            c = [pcs_grid_coords{1}(i) ; pcs_grid_coords{2}(j) ; pcs_grid_coords{3}(k)];
            x(uncut_axes) = W*(c-c0);
            X(cnt,:) = x/norm(x);
            cnt = cnt+1;
         end
      end
   end
   
else
   fprintf('Error: bingham_sample_pcs() only supports d=3\n');
   return;
end
   
   
   
   
   