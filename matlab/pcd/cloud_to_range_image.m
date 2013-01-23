function R = cloud_to_range_image(cloud, viewpoint, res, padding, blurring, blur_thresh)
%R = cloud_to_range_image(cloud, viewpoint, res, padding, blurring, blur_thresh) --
%returns (R.image, R.idx, R.min, R.res)

if nargin < 2 || isempty(viewpoint)
    viewpoint = [0,0,0];
end
if nargin < 3
    res = pi/360;
end
if nargin < 4
    padding = 0;
end
if nargin < 6
    if nargin >= 5
        fprintf('Error: must input blur_thresh if blurring = 1\n');
        return
    end
    blurring = 0;
end

origin = viewpoint(1:3);
P = cloud - repmat(origin, [size(cloud,1) 1]);
if length(viewpoint)==7
    rot_mat = quaternionToRotationMatrix(qinv(viewpoint(4:7)));
    P = P*rot_mat';
    
elseif norm(origin) > .0001  %dbug: move this outside of cloud_to_range_image!!!
    
    % get a rotation matrix that moves the cloud directly above the viewpoint (on the z-axis)
%     a = acos(origin(3));
%     if (abs(a) > .01)
%         v = cross(-origin, [0,0,1]);
%         q = [cos(a/2), sin(a/2)*v/norm(v)];
%         P = P*quaternionToRotationMatrix(q)';
%     end
    r3 = -origin';
    r3 = r3/norm(r3);
    r1 = (eye(3) - r3*r3')*rand(3,1);
    r1 = r1/norm(r1);
    r2 = cross(r3,r1);
    rot_mat = [r1,r2,r3]';  % transpose = inverse for rotation matrices
    P = P*rot_mat';
    viewpoint(4:7) = rotationMatrixToQuaternion(rot_mat);
end
D = sqrt(sum(P.^2,2));
X = atan2(P(:,1), P(:,3));
Y = acos(P(:,2)./D);

R.res = res;
R.min = [min(X) min(Y)] - R.res/2 - R.res*padding;
%R.image = -ones(ceil(.2*2*pi/R.res));
w0 = ceil(2*pi/R.res);
R.image = -ones(w0);
R.vp = viewpoint;
R.idx = zeros(w0);

w = 0;
h = 0;
for i=1:length(X)
    c = ceil(([X(i) Y(i)] - R.min)/R.res);
    if c(1) > size(R.image,1) || c(2) > size(R.image,2) || R.image(c(1),c(2)) > D(i) || R.image(c(1),c(2)) < 0
        R.image(c(1),c(2)) = D(i);
        R.idx(c(1),c(2)) = i;
        w = max(w,c(1));
        h = max(h,c(2));
    end
    if blurring
        if c(1) > 1 && (R.image(c(1)-1, c(2)) < 0 || R.image(c(1)-1,c(2)) > D(i)+blur_thresh)
            R.image(c(1)-1, c(2)) = D(i);
            R.idx(c(1)-1, c(2)) = i;
        end
        if c(2) > 1 && (R.image(c(1), c(2)-1) < 0 || R.image(c(1),c(2)-1) > D(i)+blur_thresh)
            R.image(c(1), c(2)-1) = D(i);
            R.idx(c(1), c(2)-1) = i;
        end
        if c(1)+1 > size(R.image,1) || R.image(c(1)+1, c(2)) < 0 || R.image(c(1)+1, c(2)) > D(i)+blur_thresh
            R.image(c(1)+1, c(2)) = D(i);
            R.idx(c(1)+1, c(2)) = i;
            w = max(w,c(1)+1);
        end
        if c(2)+1 > size(R.image,2) || R.image(c(1), c(2)+1) < 0 || R.image(c(1), c(2)+1) > D(i)+blur_thresh
            R.image(c(1), c(2)+1) = D(i);
            R.idx(c(1), c(2)+1) = i;
            h = max(h,c(2)+1);
        end
    end
end
w = min(w + padding, size(R.image,1));
h = min(h + padding, size(R.image,2));

%R
%max1 = max(R.image);
%max2 = max(R.image,[],2);
%w = find(max1>0, 1, 'last');
%h = find(max2>0, 1, 'last');
R.image = R.image(1:w, 1:h);
R.idx = R.idx(1:w, 1:h);

