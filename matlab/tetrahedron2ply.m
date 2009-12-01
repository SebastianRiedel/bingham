function tetrahedron2ply(x, filename)
% tetrahedron2ply(x, filename) -- Save a tetrahedron to a PLY file.

f = fopen(filename, 'w');

num_vertices = 4;
num_faces = 4;

fprintf(f, 'ply\n');
fprintf(f, 'format ascii 1.0\n');
fprintf(f, 'comment tetramesh model\n');  
fprintf(f, 'element vertex %d\n', num_vertices);
fprintf(f, 'property float x\n');
fprintf(f, 'property float y\n');
fprintf(f, 'property float z\n');
fprintf(f, 'element face %d\n', num_faces);
fprintf(f, 'property list uchar int vertex_indices\n');
fprintf(f, 'end_header\n');

for i=1:4
   fprintf(f, '%f %f %f\n', x(1,i), x(2,i), x(3,i));
end

fprintf(f, '3 %d %d %d\n', 0, 1, 2);
fprintf(f, '3 %d %d %d\n', 0, 1, 3);
fprintf(f, '3 %d %d %d\n', 0, 2, 3);
fprintf(f, '3 %d %d %d\n', 1, 2, 3);

fclose(f);
