function x = bingham_mode(B)
% x = bingham_mode(B)

if B.Z==0  %uniform
    x = rand(1,B.d);
    x = x/norm(x);
    return
end

d = B.d - 1;
V = B.V;

if d==3

    x = cross4d(V(:,1), V(:,2), V(:,3));
    
%    % search for axis to cut the hypersphere in half
%    det_V = [det(V([2 3 4],:)), det(V([1 3 4],:)), det(V([1 2 4],:)), det(V([1 2 3],:))];
%    [dmax imax] = max(abs(det_V));
%    cut_axis = imax;
%    
%    uncut_axes = [1:cut_axis-1, cut_axis+1:4];
%    c0 = V(cut_axis,:)';
%    W = inv(V(uncut_axes,:))';
%    
%    x = zeros(1,d+1);
%    x(cut_axis) = 1;
% 
%    % sample uncut coords x = W*(c-c0) and normalize
%    c = [0 ; 0 ; 0];
%    x(uncut_axes) = W*(c-c0);
%    x = x/norm(x);
   
else
   fprintf('Error: bingham_sample_pcs() only supports d=3\n');
   return;
end
