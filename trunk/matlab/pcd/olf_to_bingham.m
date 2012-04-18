function B = olf_to_bingham(nx, ny, nz, pcx, pcy, pcz, pc1, pc2)
%B = olf_to_bingham(nx, ny, nz, pcx, pcy, pcz, pc1, pc2)

B.d = 4;

r1 = [nx; ny; nz];
r2 = [pcx; pcy; pcz];
r3 = cross(r1,r2);
v1 = rotation_matrix_to_quaternion([r1,r2,r3])';  % mode
v2 = rotation_matrix_to_quaternion([r1,-r2,-r3])';  % rotation about the normal vector

% find v3 orthogonal to v1,v2
iv1 = find(v1,1);
iv2 = find(v2,1);
if iv1==iv2
    iv3 = [iv1, setdiff(1:4, [iv1,iv2])];
else
    iv3 = [iv1, iv2, setdiff(1:4, [iv1,iv2])];
end
iv3 = iv3(1:3);
v3 = [0,0,0,0]';
v3(iv3) = cross(v1(iv3), v2(iv3));
v3 = v3/norm(v3);

v4 = cross4d(v1,v2,v3);  % orthogonal v1,v2,v3
B.V = [v3,v4,v2];

fprintf('pc1 = %f, pc2 = %f, pc1/pc2 = %f\n', pc1, pc2, pc1/pc2);

z3 = min(20*pc1/pc2, 400);
B.Z = [-400, -400, -z3];

[F dF] = bingham_F(B.Z);
B.F = F;
B.dF = dF;
