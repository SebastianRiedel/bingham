function B_rot = bingham_rotate_3d(B, qv)
% B_rot = bingham_rotate_3d(B, q) -- rotate a bingham by a quaternion
% vector

B_rot = B;

B_rot.V(:,1) = quaternion_mult(qv, B.V(:,1));
B_rot.V(:,2) = quaternion_mult(qv, B.V(:,2));
B_rot.V(:,3) = quaternion_mult(qv, B.V(:,3));
