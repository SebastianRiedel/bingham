function w = crossnd(V)
% w = crossnd(V) -- Completes the orthogonal (d)x(d-1) matrix V

d = size(V,1);

I = eye(d);
w = randn(d,1);
for i=1:d-1
    w = (I - V(:,i)*V(:,i)')*w;
end
w = w/norm(w);

% search for axis to cut the hypersphere in half
% det_V = zeros(1,d-1);
% for i=1:d-1
%     det_V(i) = det(V([1:i-1, i+1:d],:));
% end
% [~, cut_axis] = max(abs(det_V));
% 
% uncut_axes = [1:cut_axis-1, cut_axis+1:d];
% c0 = V(cut_axis,:)';
% 
% w = zeros(d,1);
% w(cut_axis) = 1;
% 
% % sample uncut coords w = W*(c-c0) and normalize
% c = zeros(d-1,1);
% w(uncut_axes) = V(uncut_axes,:)' \ (c-c0);
% w = w/norm(w);
