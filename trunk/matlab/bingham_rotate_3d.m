function B_rot = bingham_rotate_3d(arg1, arg2)
% B_rot = bingham_rotate_3d(B,q) -- pre-rotate a bingham by a quaternion
% B_rot = bingham_rotate_3d(q,B) -- post-rotate a bingham by a quaternion

if isstruct(arg1)  % pre-rotate
    B = arg1;
    q = arg2;
    B_rot = B;
    B_rot.V(:,1) = quaternion_mult(B.V(:,1), q);
    B_rot.V(:,2) = quaternion_mult(B.V(:,2), q);
    B_rot.V(:,3) = quaternion_mult(B.V(:,3), q);

else  % post-rotate
    q = arg1;
    B = arg2;
    B_rot = B;
    B_rot.V(:,1) = quaternion_mult(q, B.V(:,1));
    B_rot.V(:,2) = quaternion_mult(q, B.V(:,2));
    B_rot.V(:,3) = quaternion_mult(q, B.V(:,3));    
end
