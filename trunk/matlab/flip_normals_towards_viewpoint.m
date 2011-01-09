function pcd2 = flip_normals_towards_viewpoint(pcd, vp)
%pcd2 = flip_normals_towards_viewpoint(pcd, vp)

if size(vp,2) == 1
    vp = vp';
end
VP = repmat(vp, [size(pcd.X,1),1]);

flip = sum((VP - [pcd.X, pcd.Y, pcd.Z]).*[pcd.NX, pcd.NY, pcd.NZ], 2) < 0;
C = ~flip - flip;
%C = -C;  %dbug

pcd2 = pcd;
pcd2.NX = C.*pcd.NX;
pcd2.NY = C.*pcd.NY;
pcd2.NZ = C.*pcd.NZ;

pcd2.data = populate_pcd_data(pcd2);

if isfield(pcd, 'Q')
   pcd2.Q = get_pcd_quaternions(pcd2.data, pcd2.columns);
end
