function B_rot = bingham_rotate_3d(B, qv)
% B_rot = bingham_rotate_3d(B, q) -- rotate a bingham by a quaternion
% vector

B_rot = B;

%v1 = quaternion(B.V(1,1), B.V(2,1), B.V(3,1), B.V(4,1));
%v2 = quaternion(B.V(1,2), B.V(2,2), B.V(3,2), B.V(4,2));
%v3 = quaternion(B.V(1,3), B.V(2,3), B.V(3,3), B.V(4,3));
%q = quaternion(qv(1), qv(2), qv(3), qv(4));
%qv1 = q*v1, qv2 = q*v2, qv3 = q*v3

B_rot.V(:,1) = quaternion_mult(qv, B.V(:,1));
B_rot.V(:,2) = quaternion_mult(qv, B.V(:,2));
B_rot.V(:,3) = quaternion_mult(qv, B.V(:,3));
%B_rot.V(:,1) = quaternion_mult(B.V(:,1), qv);
%B_rot.V(:,2) = quaternion_mult(B.V(:,2), qv);
%B_rot.V(:,3) = quaternion_mult(B.V(:,3), qv);

%B_rot.V
