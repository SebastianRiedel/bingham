function w = cross4d(x,y,z)
% w = cross4d(x,y,z) -- Takes the 4-D cross product of the 4x1 unit vectors x,y,z.

V = [x,y,z];

% search for axis to cut the hypersphere in half
det_V = [det(V([2 3 4],:)), det(V([1 3 4],:)), det(V([1 2 4],:)), det(V([1 2 3],:))];
[dmax imax] = max(abs(det_V));
cut_axis = imax;

uncut_axes = [1:cut_axis-1, cut_axis+1:4];
c0 = V(cut_axis,:)';
W = inv(V(uncut_axes,:))';

w = zeros(4,1);
w(cut_axis) = 1;

% sample uncut coords w = W*(c-c0) and normalize
c = [0 ; 0 ; 0];
w(uncut_axes) = W*(c-c0);
w = w/norm(w);
