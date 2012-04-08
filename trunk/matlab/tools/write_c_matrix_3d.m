function write_c_matrix_3d(f, X)

n1 = size(X,1);
n2 = size(X,2);

fprintf(f, '{\n');
for i=1:n1
   fprintf(f, '{\n');
   for j=1:n2
      fprintf(f, '{');
      fprintf(f, '%f', X(i,j,1));
      fprintf(f, ', %f', X(i,j,2:end));
      fprintf(f, '}');
      if j < n2
         fprintf(f, ',');
      end
      fprintf(f, '\n');
   end
   fprintf(f, '}');
   if i < n1
      fprintf(f, ',');
   end
   fprintf(f, '\n');
end
fprintf(f, '};\n');
