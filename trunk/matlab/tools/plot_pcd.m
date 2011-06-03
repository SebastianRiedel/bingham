function plot_pcd(pcd, style, downsample)
% plot_pcd(pcd)

if nargin < 2
    style = '.';
end

if nargin < 3
    X = pcd.X;
    Y = pcd.Y;
    Z = pcd.Z;
else
    step = ceil(1/downsample);
    X = pcd.X(1:step:end);
    Y = pcd.Y(1:step:end);
    Z = pcd.Z(1:step:end);
end

plot3(X, Y, Z, style); %, 'MarkerSize', 20);
axis vis3d;
axis equal;

