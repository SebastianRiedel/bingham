function R = cloud_to_range_image(cloud, viewpoint, res)
% R = cloud_to_range_image(cloud, viewpoint) -- returns (R.image, R.min, R.res)

if nargin < 3
    res = pi/360;
end

origin = viewpoint(1:3);
P = cloud - repmat(origin, [size(cloud,1) 1]);
D = sqrt(sum(P.^2,2));
X = atan2(P(:,1), P(:,3));
Y = asin(P(:,2)./D);

R.res = res;
R.min = [min(X) min(Y)] - R.res/2;
R.image = -ones(ceil(.1*2*pi/R.res));

w = 0;
h = 0;
for i=1:length(X)
    c = ceil(([X(i) Y(i)] - R.min)/R.res);
    if c(1) > size(R.image,1) || c(2) > size(R.image,2) || R.image(c(1),c(2)) > D(i) || R.image(c(1),c(2)) < 0
        R.image(c(1),c(2)) = D(i);
        w = max(w,c(1));
        h = max(h,c(2));
    end
end

%R
%max1 = max(R.image);
%max2 = max(R.image,[],2);
%w = find(max1>0, 1, 'last');
%h = find(max2>0, 1, 'last');
R.image = R.image(1:h, 1:w);

