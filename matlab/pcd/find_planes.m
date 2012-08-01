function [coeffs, inliers] = find_planes(cloud)
%[coeffs, inliers] = find_planes(cloud)

num_points = size(cloud,1);
min_inliers = num_points/30;
inlier_dist = .01;
num_samples = 500;

coeffs = [];
inliers = {};
cnt = 0;
index_map = 1:num_points;

while 1
    num_points = size(cloud,1);
    hcloud = [cloud, ones(num_points,1)];
    fprintf('%d\n', num_points);
    cmax = [];
    nmax = 0;
    for i=1:num_samples
        % sample 3 points
        point_indices = ceil(rand(1,3)*num_points);
        while length(unique(point_indices)) < 3
            point_indices = ceil(rand(1,3)*num_points);
        end
        points = cloud(point_indices,:);
        p1 = points(1,:);
        p2 = points(2,:);
        p3 = points(3,:);
        
        % compute plane coeffs
        n = cross(p2-p1, p3-p1);
        n = n/norm(n);
        c = [n, -dot(n,p1)];

        % compute num inliers
        C = repmat(c, [num_points,1]);
        n = sum(abs(sum(C.*hcloud,2)) < inlier_dist);
        if n > nmax
            nmax = n;
            cmax = c;
        end
    end
    
    if nmax >= min_inliers
        cnt = cnt+1;
        coeffs(cnt,:) = cmax;
        C = repmat(cmax, [num_points,1]);
        inlier_mask = abs(sum(C.*hcloud,2)) < inlier_dist;
        inliers{cnt} = index_map(inlier_mask);
        cloud = cloud(~inlier_mask,:);
        index_map = index_map(~inlier_mask);
    else
        break;
    end
end



