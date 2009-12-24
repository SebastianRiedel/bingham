function pcd2 = rotate_pcd(pcd, q)
% pcd2 = rotate_pcd(pcd, q)

pcd2 = pcd;

R = quaternionToRotationMatrix(q);

V = R*[pcd.X' ; pcd.Y' ; pcd.Z'];
pcd2.X = V(1,:)';
pcd2.Y = V(2,:)';
pcd2.Z = V(3,:)';

if isfield(pcd, 'NX')
    V = R*[pcd.NX' ; pcd.NY' ; pcd.NZ'];
    pcd2.NX = V(1,:)';
    pcd2.NY = V(2,:)';
    pcd2.NZ = V(3,:)';
end

if isfield(pcd, 'PCX')
    V = R*[pcd.PCX' ; pcd.PCY' ; pcd.PCZ'];
    pcd2.PCX = V(1,:)';
    pcd2.PCY = V(2,:)';
    pcd2.PCZ = V(3,:)';
end

%pcd2.Q = get_pcd_quaternions
