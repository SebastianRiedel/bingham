function pcd2 = shift_pcd(pcd, x)
% pcd2 = shift_pcd(pcd, x)

pcd2 = pcd;

pcd2.X = pcd2.X + x(1);
pcd2.Y = pcd2.Y + x(2);
pcd2.Z = pcd2.Z + x(3);

pcd2.data = populate_pcd_data(pcd2);
