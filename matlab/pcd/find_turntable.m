function [x,n] = find_turntable(pcd)
%[x,n] = find_turntable(pcd) -- returns the centroid and normal

% downsample and remove NaN
cloud = [pcd.X, pcd.Y, pcd.Z];
cloud = cloud(1:10:end, :);
cloud = cloud(~max(isnan(cloud),[],2),:);

% find all the planes
[coeffs, inliers] = find_planes(cloud);

% find the turntable plane
max_theta = pi/10;
C = repmat(coeffs(1,1:3), [size(coeffs,1),1]);
dtheta = abs(acos(sum(C.*coeffs(:,1:3),2)));
dtheta = min(dtheta, pi-dtheta);
coeffs(dtheta > max_theta, 4) = inf;
[xxx, iturn] = min(abs(coeffs(:,4)));

% get turntable normal and centroid
n = coeffs(iturn,1:3);
W = exp(-sum(cloud(inliers{iturn},:).*cloud(inliers{iturn},:), 2));
W = W./sum(W);
x = sum(repmat(W, [1 3]).*cloud(inliers{iturn},:));

% refine normal estimate
DX = cloud(inliers{iturn},:) - repmat(x, [length(inliers{iturn}),1]);
I = inliers{iturn}( sum(DX.*DX, 2) < .25 );
C = princomp(cloud(I,:));
n = C(:,3)';
if dot(x,n) > 0
    n = -n;
end

figure(1);
clf;
hold on;
for i=1:size(coeffs,1)
    if i==iturn
        color = [1,0,0];
    else
        color = rand()*[1,1,1];
    end
    DX = cloud(inliers{i},:) - repmat(x, [length(inliers{i}),1]);
    I = inliers{i}( sum(DX.*DX, 2) < .25 );
    plot3(cloud(I,1), cloud(I,2), cloud(I,3), '.', 'Color', color);
end
plot3([x(1); x(1)+.1*n(1)], [x(2); x(2)+.1*n(2)], [x(3); x(3)+.1*n(3)], 'bo-', 'MarkerSize', 10, 'LineWidth', 5);
hold off;
axis vis3d;
axis equal;
drawnow;
%input(':');

