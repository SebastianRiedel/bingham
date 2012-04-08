function pcd2 = transform_pcd(pcd, x, q)
% pcd2 = transform_pcd(pcd, x, q) -- rotate by q, then shift by x

pcd2 = rotate_pcd(pcd, q);
pcd2 = shift_pcd(pcd2, x);
