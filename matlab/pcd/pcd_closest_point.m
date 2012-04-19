function i = pcd_closest_point(p, pcd, I)
%i = pcd_closest_point(p, pcd)
%i = pcd_closest_point(p, pcd, I)

% make p a column vector
if size(p,1)==1
    p = p';
end

if nargin < 3
    I = [];
end

if isempty(I)
    cloud = [pcd.X, pcd.Y, pcd.Z];
else
    cloud = [pcd.X(I), pcd.Y(I), pcd.Z(I)];
end

n = size(cloud,1);
D2 = sum((repmat(p, [1,n]) - cloud').^2);

[~,i] = min(D2);

if ~isempty(I)
    i = I(i);
end
