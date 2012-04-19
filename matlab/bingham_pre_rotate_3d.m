function B_rot = bingham_pre_rotate_3d(B, qv)
% B_rot = bingham_pre_rotate_3d(B, q) -- pre-rotate a bingham by a quaternion
% vector

B_rot = B;

B_rot.V(:,1) = quaternion_mult(B.V(:,1), qv);
B_rot.V(:,2) = quaternion_mult(B.V(:,2), qv);
B_rot.V(:,3) = quaternion_mult(B.V(:,3), qv);
