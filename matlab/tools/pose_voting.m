function pose_voting(Q, B, C, q_obs, f_obs)
% pose_voting(Q, B, C, pcd) -- computes a posterior distribution over
% quaternions, Q, given bingham mixtures, B, cluster feature centers, C,
% and point cloud, pcd.

n = size(pcd.data, 1);
k = length(B);

for i=1:size(Q,1)   % cells
    
    % rotate the bingham mixture distributions by Q(i,:)
    for c=1:k
        for j=1:length(B{c})
            B_rot{c}(j) = bingham_rotate_3d(B{c}(j), Q(i,:));
        end
    end
    
    for j=



