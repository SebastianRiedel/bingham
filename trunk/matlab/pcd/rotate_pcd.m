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

if isfield(pcd, 'Q')
    for i=1:size(pcd.Q,1)
        pcd2.Q(i,:,1) = quaternion_mult(q, pcd.Q(i,:,1));
        pcd2.Q(i,:,2) = quaternion_mult(q, pcd.Q(i,:,2));
    end
end

if isfield(pcd, 'vp') && ~isempty(pcd.vp)
   pcd2.vp(1:3) = pcd.vp(1:3)*R';
   pcd2.vp(4:7) = quaternion_mult(q, pcd.vp(4:7));
end

pcd2.data = populate_pcd_data(pcd2);
