function I = range_image_xyz2ind(RI, cloud)
%I = range_image_xyz2ind(RI, cloud)

viewpoint = RI.vp;
origin = viewpoint(1:3);
P = cloud - repmat(origin, [size(cloud,1) 1]);

if length(viewpoint)==7
    rot_mat = quaternionToRotationMatrix(qinv(viewpoint(4:7)));
    P = P*rot_mat';
end

D = sqrt(sum(P.^2,2));
X = atan2(P(:,1), P(:,3));
Y = acos(P(:,2)./D);

C = ceil(([X,Y] - repmat(RI.min, [length(X),1]))/RI.res);
mask = logical((C(:,1)>0).*(C(:,2)>0).*(C(:,1)<=size(RI.image,1)).*(C(:,2)<=size(RI.image,2)));

I = zeros(length(X),1);
I(mask) = sub2ind(size(RI.image), C(mask,1), C(mask,2));
