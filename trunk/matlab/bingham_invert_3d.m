function B_inv = bingham_invert_3d(B)
% B_inv = bingham_invert_3d(B) -- invert a quaternion Bingham distribution

B_inv = B;

H = diag([1,-1,-1,-1]);
B_inv.V = H*B.V;
